"""Mode A: Live Groove - BPM-driven dance mode using microphone input.

Listens to live audio via the robot's USB microphone, detects BPM,
and executes pre-recorded dance moves from the library.

This is the "Listener" - social, present, organic. Designed to exist
in a room with people and account for the messiness of the real world.

Key features:
- Noise calibration (captures motor noise profile during breathing/dancing)
- Spectral noise subtraction to filter out motor sounds
- Librosa BPM detection with half/double-time clamping
- BPM stability tracking: Gathering → Locked → Unstable
- Pre-recorded library moves (reachy_mini_dances_library)
- Breathing idle motion when no music detected
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import librosa  # type: ignore
import numpy as np  # type: ignore
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES  # type: ignore

from reachy_mini import ReachyMini  # type: ignore

from .. import mode_settings, move_config
from ..core.safety_mixer import MovementIntent
from .base import DanceMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


# Initialize move_config with available moves
move_config.init_moves(list(AVAILABLE_MOVES.keys()))


# Per-move amplitude scaling (1.0 = full, 0.8 = 20% reduction, etc.)
# Moves not listed here use 1.0 (full amplitude)
MOVE_AMPLITUDE_OVERRIDES = {
    "headbanger_combo": 0.3,
    "dizzy_spin": 0.8,
    "pendulum_swing": 0.4,
    "jackson_square": 0.6,
    "side_to_side_sway": 0.65,
    "sharp_side_tilt": 0.35,
    "grid_snap": 0.5,
    "side_peakaboo": 0.4,
    "simple_nod": 0.5,
    "chin_lead": 0.9,
}


@dataclass
class LiveGrooveConfig:
    """Configuration for Live Groove mode."""

    # Main control loop period (s)
    control_ts: float = 0.01

    # Audio
    audio_rate: int = 16000
    audio_chunk_size: int = 2048
    audio_win: float = 1.6  # Analysis window in seconds

    # Noise calibration
    silence_calibration_duration: float = 4.0  # Background silence phase
    noise_calibration_duration: float = 14.0  # Breathing phase
    dance_noise_calibration_duration: float = 8.0  # Dance phase
    noise_subtraction_strength: float = 1.0

    # BPM detection
    bpm_min: float = 70.0
    bpm_max: float = 140.0
    bpm_stability_buffer: int = 6
    bpm_stability_threshold: float = 5.0

    # Timing
    silence_tmo: float = 2.0
    volume_gate_threshold: float = 0.005  # Lowered from 0.008 for better sensitivity
    music_confidence_ratio: float = 1.5  # Signal must be 1.5x threshold to dance
    beats_per_sequence: int = 4
    min_breathing_between_moves: float = 0
    unstable_periods_before_stop: int = 4

    # Neutral pose
    neutral_pos: np.ndarray[Any, np.dtype[np.float64]] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.01])
    )
    neutral_eul: np.ndarray[Any, np.dtype[np.float64]] = field(
        default_factory=lambda: np.zeros(3)
    )

    # Derived
    audio_buffer_len: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate buffer length from rate and window."""
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


class MusicState:
    """Thread-safe state for audio analysis."""

    def __init__(self) -> None:
        """Initialize thread-safe state."""
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = "Init"
        self.beats: collections.deque[float] = collections.deque(maxlen=512)
        self.unstable_period_count = 0
        self.has_ever_locked = False
        self.bpm_std = 0.0
        self.cleaned_amplitude = 0.0
        self.raw_amplitude = 0.0
        self.is_breathing = True
        self.music_confident = False  # True when cleaned signal is well above threshold


def compute_noise_profile(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    method: str = "median",
    exclude_transients: bool = False,
    transient_threshold: float = 2.0,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float, dict[str, Any]]:
    """Compute spectral noise profile from audio samples.

    Uses median (robust to outliers) and can exclude transient frames (collisions/knocks).

    Args:
        audio: Raw audio samples
        method: "mean", "median", or "percentile_25"
        exclude_transients: If True, detect and exclude collision/impact frames
        transient_threshold: Frames with RMS > threshold * median_rms are excluded

    Returns:
        (profile, rms, stats) where stats contains analysis metadata

    """
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    # Compute per-frame RMS to detect transients
    frame_rms = np.sqrt(np.mean(magnitude**2, axis=0))
    median_frame_rms = np.median(frame_rms)

    stats = {
        "total_frames": magnitude.shape[1],
        "excluded_frames": 0,
        "transient_threshold": transient_threshold,
        "median_frame_rms": float(median_frame_rms),
        "method": method,
    }

    if exclude_transients and median_frame_rms > 0:
        # Find frames that are likely transients (collisions, knocks)
        threshold = median_frame_rms * transient_threshold
        quiet_mask = frame_rms < threshold

        stats["excluded_frames"] = int(np.sum(~quiet_mask))
        stats["max_frame_rms"] = float(np.max(frame_rms))

        if np.sum(quiet_mask) > 10:  # Need at least some quiet frames
            magnitude = magnitude[:, quiet_mask]
            if stats["excluded_frames"] > 0:
                logger.debug(
                    f"      Excluded {stats['excluded_frames']} transient frames"
                )

    # Compute profile using selected method
    if method == "median":
        profile = np.median(magnitude, axis=1)
    elif method == "percentile_25":
        profile = np.percentile(magnitude, 25, axis=1)
    else:  # mean
        profile = np.mean(magnitude, axis=1)

    rms = float(np.sqrt(np.mean(audio**2)))
    return profile, rms, stats


def load_environment_profile(
    profile_path: str | Path,
) -> (
    tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ]
    | None
):
    """Load saved environment profile from .npz file.

    Returns:
        Tuple of (silence_profile, breathing_profile, dance_profile) or None if loading fails.

    """
    path = Path(profile_path)
    if not path.exists():
        logger.warning(f"[LiveGroove] WARNING: Profile file not found: {path}")
        return None

    try:
        data = np.load(path, allow_pickle=True)
        silence_profile = data["silence_profile"]
        breathing_profile = data["breathing_profile"]
        dance_profile = data["dance_profile"]

        # Parse metadata for display
        metadata = json.loads(str(data["metadata"]))
        logger.info(f"[LiveGroove] Loaded profile from {path}")
        logger.info(f"   Created: {metadata.get('created', 'unknown')}")
        logger.info(f"   Silence RMS: {metadata.get('silence_rms', 0):.6f}")
        logger.info(f"   Breathing RMS: {metadata.get('breathing_rms', 0):.6f}")
        logger.info(f"   Dance RMS: {metadata.get('dance_rms', 0):.6f}")

        return silence_profile, breathing_profile, dance_profile

    except Exception as e:
        logger.warning(f"[LiveGroove] WARNING: Failed to load profile: {e}")
        return None


class MoveChoreographer:
    """Manages dance move selection and sequencing."""

    def __init__(self) -> None:
        """Initialize choreographer with available moves."""
        self.base_moves = list(AVAILABLE_MOVES.keys())
        self.move_names: list[str] = []
        self.waveforms = ["sin"]
        self.move_idx = 0
        self.waveform_idx = 0
        self.amplitude_scale = 1.0
        self.beat_counter_for_cycle = 0.0
        # Build initial move list with mirrored versions
        self.rebuild_move_list()

    def rebuild_move_list(self) -> None:
        """Rebuild move list including enabled mirrored versions."""
        moves = list(self.base_moves)

        # Add mirrored versions for moves that have mirror enabled
        mirror_settings = move_config.get_all_mirror()
        for move_name, is_mirrored in mirror_settings.items():
            if is_mirrored and move_name in self.base_moves:
                moves.append(f"{move_name}_mirrored")

        random.shuffle(moves)
        self.move_names = moves
        self.move_idx = 0
        logger.info(
            f"[MoveChoreographer] Rebuilt move list: {len(moves)} moves ({len(self.base_moves)} base + {len(moves) - len(self.base_moves)} mirrored)"
        )

    def current_move_name(self) -> str:
        """Get the name of the currently active move."""
        return self.move_names[self.move_idx]

    def current_waveform(self) -> str:
        """Get the currently selected waveform."""
        return self.waveforms[self.waveform_idx]

    def advance_move(self) -> None:
        """Advance to next move."""
        self.move_idx = (self.move_idx + 1) % len(self.move_names)
        if self.move_idx == 0:
            random.shuffle(self.move_names)
        self.beat_counter_for_cycle = 0.0

    def request_move(self, move_name: str) -> bool:
        """Request a specific move immediately."""
        if move_name in self.move_names:
            self.move_idx = self.move_names.index(move_name)
            self.beat_counter_for_cycle = 0.0
            return True
        elif f"{move_name}_mirrored" in self.move_names:
            self.move_idx = self.move_names.index(f"{move_name}_mirrored")
            self.beat_counter_for_cycle = 0.0
            return True
        return False


class LiveGroove(DanceMode):
    """Live Groove: Real-time BPM-driven dancing with pre-recorded moves."""

    MODE_ID = "live_groove"
    MODE_NAME = "Live Groove"

    # Default profile location
    DEFAULT_PROFILE_PATH = Path(__file__).parent.parent / "environment_profile.npz"

    def __init__(
        self,
        safety_mixer: SafetyMixer,
        mini: ReachyMini,
        profile_path: Optional[str] = None,
        skip_calibration: bool = False,
        force_calibration: bool = False,
    ) -> None:
        """Initialize Live Groove mode."""
        super().__init__(safety_mixer)

        self.mini = mini  # Robot instance for audio access
        self.config = LiveGrooveConfig()

        # Load dynamic settings from file/global store
        settings = mode_settings.get_mode_settings(self.MODE_ID)
        if "volume_gate_threshold" in settings:
            self.config.volume_gate_threshold = settings["volume_gate_threshold"]
        if "bpm_stability_threshold" in settings:
            self.config.bpm_stability_threshold = settings["bpm_stability_threshold"]

        self.music_state = MusicState()
        self.choreographer = MoveChoreographer()

        # Profile settings
        self.profile_path = profile_path
        self.skip_calibration = skip_calibration
        self.force_calibration = force_calibration  # If True, ignore default profile

        # Noise profiles (3-phase)
        self.silence_noise_profile: Optional[np.ndarray[Any, np.dtype[np.float32]]] = (
            None
        )
        self.breathing_noise_profile: Optional[
            np.ndarray[Any, np.dtype[np.float32]]
        ] = None
        self.dance_noise_profile: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None

        # Threading
        self.stop_event = threading.Event()
        self.audio_thread: Optional[threading.Thread] = None
        self.control_thread: Optional[threading.Thread] = None

        # Status
        self._status = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "bpm": 0.0,
            "move": "",
            "calibrated": False,
            "music_confident": False,
            "volume_threshold": self.config.volume_gate_threshold,
        }

        # Load settings from mode_settings
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("live_groove")
        # Live Groove's intensity is applied via SafetyMixer
        if "intensity" in settings:
            self.mixer.update_config(intensity=settings["intensity"])
        # Load volume threshold if specified
        if "volume_gate_threshold" in settings:
            self.config.volume_gate_threshold = settings["volume_gate_threshold"]
            self._status["volume_threshold"] = settings["volume_gate_threshold"]

    def apply_settings(self, updates: dict[str, float]) -> None:
        """Apply settings updates (called from API for live tuning)."""
        if "intensity" in updates:
            self.mixer.update_config(intensity=updates["intensity"])
        if "volume_gate_threshold" in updates:
            self.config.volume_gate_threshold = updates["volume_gate_threshold"]
            self._status["volume_threshold"] = updates["volume_gate_threshold"]
            logger.info(
                f"[{self.MODE_NAME}] Volume threshold updated to {updates['volume_gate_threshold']:.4f}"
            )
        if "bpm_stability_threshold" in updates:
            self.config.bpm_stability_threshold = updates["bpm_stability_threshold"]
            logger.info(
                f"[{self.MODE_NAME}] BPM stability threshold updated to {updates['bpm_stability_threshold']:.2f}"
            )

    def refresh_moves(self) -> None:
        """Rebuild the move list with current mirror settings."""
        self.choreographer.rebuild_move_list()

    async def start(self) -> None:
        """Start Live Groove mode."""
        if self.running:
            return

        logger.info(f"[{self.MODE_NAME}] Starting...")

        # Try to load saved profile, otherwise run live calibration
        profile_loaded = False

        if self.profile_path:
            # Explicit profile path provided
            logger.info(
                f"[{self.MODE_NAME}] Loading explicit profile: {self.profile_path}"
            )
            loaded = load_environment_profile(self.profile_path)
            if loaded:
                (
                    self.silence_noise_profile,
                    self.breathing_noise_profile,
                    self.dance_noise_profile,
                ) = loaded
                profile_loaded = True
                logger.info(f"[{self.MODE_NAME}] Using saved environment profile!")
            else:
                logger.info(f"[{self.MODE_NAME}] Failed to load explicit profile.")
        elif self.DEFAULT_PROFILE_PATH.exists() and not self.force_calibration:
            # Try default profile (unless force_calibration is True)
            logger.info(
                f"[{self.MODE_NAME}] Found default profile at: {self.DEFAULT_PROFILE_PATH}"
            )
            loaded = load_environment_profile(self.DEFAULT_PROFILE_PATH)
            if loaded:
                (
                    self.silence_noise_profile,
                    self.breathing_noise_profile,
                    self.dance_noise_profile,
                ) = loaded
                profile_loaded = True
                logger.info(f"[{self.MODE_NAME}] Using default environment profile!")
            else:
                logger.info(
                    f"[{self.MODE_NAME}] Failed to load default profile from disk."
                )
        else:
            logger.info(
                f"[{self.MODE_NAME}] No default profile found at: {self.DEFAULT_PROFILE_PATH}"
            )

        if not profile_loaded and not self.skip_calibration:
            # Run calibration (this takes ~26 seconds)
            if self.force_calibration:
                logger.info(f"[{self.MODE_NAME}] Force calibration requested...")
            logger.info(f"[{self.MODE_NAME}] Running 3-phase noise calibration...")
            self._status["state"] = "calibrating"
            await self._run_calibration()
        elif self.skip_calibration:
            logger.warning(
                f"[{self.MODE_NAME}] WARNING: Skipping calibration - may dance to motor noise!"
            )

        self._status["calibrated"] = True

        # Start threads
        self.stop_event.clear()
        self.running = True

        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()

        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        self._status["running"] = True
        self._status["state"] = "listening"
        logger.info(f"[{self.MODE_NAME}] Started - play music!")

    async def stop(self) -> None:
        """Stop Live Groove mode."""
        if not self.running:
            return

        logger.info(f"[{self.MODE_NAME}] Stopping...")
        self.running = False
        self.stop_event.set()

        try:
            # Wait for threads (they handle their own cleanup in finally blocks)
            # Short timeouts to ensure we don't block app shutdown
            if self.audio_thread:
                self.audio_thread.join(timeout=0.5)
                if self.audio_thread.is_alive():
                    logger.warning(
                        f"[{self.MODE_NAME}] Warning: audio thread didn't stop cleanly"
                    )
            if self.control_thread:
                self.control_thread.join(timeout=0.5)
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Error joining threads: {e}")
        finally:
            # Always return to neutral, even if threads crash
            try:
                if hasattr(self, "mixer"):
                    self.mixer.reset()
            except Exception as e:
                logger.warning(
                    f"[{self.MODE_NAME}] Warning: Failed to reset mixer: {e}"
                )

        self._status["running"] = False
        self._status["state"] = "idle"
        logger.info(f"[{self.MODE_NAME}] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        with self.music_state.lock:
            self._status["bpm"] = self.music_state.librosa_bpm
            self._status["state"] = self.music_state.state
            self._status["music_confident"] = self.music_state.music_confident
            self._status["has_ever_locked"] = self.music_state.has_ever_locked
            self._status["raw_amplitude"] = self.music_state.raw_amplitude
        self._status["move"] = self.choreographer.current_move_name()
        self._status["volume_threshold"] = self.config.volume_gate_threshold

        # Convert any numpy types to Python native types for JSON serialization
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, "item"):  # numpy scalar
                status[key] = value.item()
        return status

    async def _run_calibration(self) -> None:
        """Run 3-phase noise calibration sequence with threaded audio capture."""
        logger.info(f"\n{'=' * 60}")
        logger.info("STARTING 3-PHASE NOISE CALIBRATION")
        logger.info("Please ensure NO MUSIC is playing during calibration!")
        logger.info(f"{'=' * 60}")

        # Phase 1: Background silence (robot still)
        logger.info(
            f"\n[{self.MODE_NAME}] PHASE 1: Background Silence ({self.config.silence_calibration_duration}s)"
        )
        logger.info("   Robot is still. Please ensure room is quiet.")
        self.silence_noise_profile = await self._calibrate_silence()

        # Phase 2: Breathing motion noise
        logger.info(
            f"\n[{self.MODE_NAME}] PHASE 2: Breathing Motor Noise ({self.config.noise_calibration_duration}s)"
        )
        logger.info("   Robot will do smooth breathing motion.")
        self.breathing_noise_profile = await self._calibrate_with_motion(
            duration=self.config.noise_calibration_duration, movement_type="breathing"
        )

        # Phase 3: Dance motion noise
        dance_amplitude = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 0.3)
        logger.info(
            f"\n[{self.MODE_NAME}] PHASE 3: Dance Motor Noise ({self.config.dance_noise_calibration_duration}s)"
        )
        logger.info(
            f"   Robot will do headbanger combo at {dance_amplitude:.0%} amplitude."
        )
        self.dance_noise_profile = await self._calibrate_with_motion(
            duration=self.config.dance_noise_calibration_duration,
            movement_type="dancing",
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("All calibration phases complete!")
        logger.info(f"{'=' * 60}\n")

        # Auto-Save Profile
        try:
            silence_rms = float(np.sqrt(np.mean(self.silence_noise_profile**2)))
            breathing_rms = float(np.sqrt(np.mean(self.breathing_noise_profile**2)))
            dance_rms = (
                float(np.sqrt(np.mean(self.dance_noise_profile**2)))
                if self.dance_noise_profile is not None
                else 0.0
            )

            metadata = {
                "created": datetime.now().isoformat(),
                "silence_rms": silence_rms,
                "breathing_rms": breathing_rms,
                "dance_rms": dance_rms,
            }

            np.savez(
                self.DEFAULT_PROFILE_PATH,
                silence_profile=self.silence_noise_profile,
                breathing_profile=self.breathing_noise_profile,
                dance_profile=self.dance_noise_profile
                if self.dance_noise_profile is not None
                else self.breathing_noise_profile,
                metadata=json.dumps(metadata),
            )
            logger.info(
                f"[{self.MODE_NAME}] Auto-saved calibration profile to {self.DEFAULT_PROFILE_PATH}"
            )
        except Exception as e:
            logger.warning(
                f"[{self.MODE_NAME}] WARNING: Failed to auto-save profile: {e}"
            )

    def _audio_capture_thread(
        self,
        duration: float,
        collected: list[np.ndarray[Any, np.dtype[np.float32]]],
        stop_event: threading.Event,
    ) -> None:
        """Background thread to capture audio without blocking motion using SDK."""
        try:
            self.mini.media.start_recording()
        except Exception as e:
            logger.error(
                f"[{self.MODE_NAME}] ERROR: Calibration failed to start recording: {e}"
            )
            return

        start_time = time.time()

        try:
            while time.time() - start_time < duration and not stop_event.is_set():
                sample = self.mini.media.get_audio_sample()

                if sample is not None:
                    sample = cast(np.ndarray[Any, np.dtype[np.float32]], sample)
                    # SDK returns stereo float32 with shape (n, 2)
                    # Handle both stereo and mono inputs
                    if sample.ndim == 2 and sample.shape[1] >= 2:
                        mono = (sample[:, 0] + sample[:, 1]) / 2.0
                    elif sample.ndim == 2 and sample.shape[1] == 1:
                        mono = sample[:, 0]
                    elif sample.ndim == 1:
                        mono = sample
                    else:
                        continue
                    collected.append(mono.astype(np.float32))

                time.sleep(0.01)  # Poll every 10ms
        finally:
            try:
                self.mini.media.stop_recording()
            except Exception as e:
                logger.warning(
                    f"[{self.MODE_NAME}] Warning: Error stopping calibration recording: {e}"
                )

    async def _calibrate_silence(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Record background silence with robot completely still."""
        duration = self.config.silence_calibration_duration

        # Move robot to neutral and hold still
        intent = MovementIntent(
            position=self.config.neutral_pos.copy(),
            orientation=self.config.neutral_eul.copy(),
            antennas=np.zeros(2),
        )
        self.mixer.send_intent(intent)
        await asyncio.sleep(0.5)  # Let robot settle

        # Start audio capture in background thread
        collected: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            args=(duration, collected, stop_event),
            daemon=True,
        )
        audio_thread.start()

        # Wait and show progress (robot stays still)
        start_time = time.time()
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            logger.debug(f"   Recording silence: {remaining:.1f}s remaining...")
            await asyncio.sleep(0.1)

        # Wait for audio thread to finish
        stop_event.set()
        audio_thread.join(timeout=2.0)

        if not collected:
            logger.warning(
                f"[{self.MODE_NAME}] WARNING: No audio collected during silence calibration!"
            )
            return np.zeros(
                1025, dtype=np.float32
            )  # Default profile shape for n_fft=2048

        audio = cast(np.ndarray[Any, np.dtype[np.float32]], np.concatenate(collected))
        profile, rms, _ = compute_noise_profile(
            audio, method="median", exclude_transients=False
        )
        logger.info(f"   Phase 1 complete. Silence RMS: {rms:.6f}")
        return profile

    async def _calibrate_with_motion(
        self, duration: float, movement_type: str
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Record noise profile while robot moves. Audio runs in separate thread."""
        # Start audio capture in background thread
        collected: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        stop_event = threading.Event()
        audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            args=(duration, collected, stop_event),
            daemon=True,
        )
        audio_thread.start()

        # Run motion in main thread (smooth updates)
        motion_time = 0.0
        start_time = time.time()
        last_loop_time = start_time

        while time.time() - start_time < duration:
            loop_start = time.time()
            dt = loop_start - last_loop_time
            last_loop_time = loop_start

            remaining = duration - (loop_start - start_time)
            phase_name = "breathing" if movement_type == "breathing" else "dance"
            logger.debug(
                f"   Recording {phase_name} noise: {remaining:.1f}s remaining..."
            )

            motion_time += dt

            if movement_type == "breathing":
                intent = self._compute_breathing_pose(motion_time)
            else:
                # Use headbanger with SAME amplitude as actual dancing (0.3)
                intent = self._compute_calibration_dance_pose(motion_time)

            self.mixer.send_intent(intent)

            # Maintain smooth control loop
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0.001, self.config.control_ts - elapsed_loop)
            await asyncio.sleep(sleep_time)

        # Stop audio capture and wait
        stop_event.set()
        audio_thread.join(timeout=2.0)

        # Return to neutral
        self.mixer.reset()

        if not collected:
            logger.warning(
                f"[{self.MODE_NAME}] WARNING: No audio collected during {movement_type} calibration!"
            )
            return (
                self.silence_noise_profile
                if self.silence_noise_profile is not None
                else np.zeros(1025, dtype=np.float32)
            )

        audio = cast(np.ndarray[Any, np.dtype[np.float32]], np.concatenate(collected))

        # Use median, no transient exclusion (capture all motor noise)
        profile, rms, _ = compute_noise_profile(
            audio, method="median", exclude_transients=False
        )

        # Combine with silence profile: take the maximum of both
        if self.silence_noise_profile is not None:
            noise_profile = cast(
                np.ndarray[Any, np.dtype[np.float32]],
                np.maximum(profile, self.silence_noise_profile),
            )
        else:
            noise_profile = profile

        phase_num = 2 if movement_type == "breathing" else 3
        logger.info(
            f"   Phase {phase_num} complete. {movement_type.capitalize()} RMS: {rms:.6f}"
        )
        return noise_profile

    def _compute_calibration_dance_pose(self, t_beats: float) -> MovementIntent:
        """Compute dance pose for calibration using reduced amplitude."""
        move_fn, base_params, _ = AVAILABLE_MOVES["headbanger_combo"]
        params = base_params.copy()

        # Use SAME amplitude as actual dancing to avoid collisions
        amp_scale = MOVE_AMPLITUDE_OVERRIDES.get("headbanger_combo", 0.3)

        # Run at 120 BPM
        t = t_beats * (120.0 / 60.0)
        offsets = move_fn(t, **params)

        return MovementIntent(
            position=self.config.neutral_pos + offsets.position_offset * amp_scale,
            orientation=self.config.neutral_eul
            + offsets.orientation_offset * amp_scale,
            antennas=offsets.antennas_offset * amp_scale,
        )

    def _compute_breathing_pose(self, t: float) -> MovementIntent:
        """Compute breathing/idle pose."""
        # Y sway
        y_amplitude = 0.016
        y_freq = 0.2
        y_offset = y_amplitude * np.sin(2.0 * np.pi * y_freq * t)

        # Head roll
        roll_amplitude = 0.222
        roll_freq = 0.15
        roll_offset = roll_amplitude * np.sin(2.0 * np.pi * roll_freq * t)

        return MovementIntent(
            position=self.config.neutral_pos + np.array([0.0, y_offset, 0.0]),
            orientation=self.config.neutral_eul + np.array([roll_offset, 0.0, 0.0]),
            antennas=np.array([-0.15, 0.15]),
        )

    def _compute_dance_pose(
        self, t_beats: float, move_name: str, bpm: float
    ) -> MovementIntent:
        """Compute dance pose from library move."""
        # Check if this is a mirrored version
        is_mirrored = move_name.endswith("_mirrored")
        base_move_name = (
            move_name[:-9] if is_mirrored else move_name
        )  # Strip "_mirrored"

        move_fn, base_params, _ = AVAILABLE_MOVES[base_move_name]
        params = base_params.copy()

        if "waveform" in params:
            params["waveform"] = self.choreographer.current_waveform()

        offsets = move_fn(t_beats, **params)
        amp_scale = move_config.get_dampening(base_move_name)

        # Apply amplitude scaling
        pos_offset = offsets.position_offset * amp_scale
        ori_offset = offsets.orientation_offset * amp_scale
        ant_offset = offsets.antennas_offset * amp_scale

        # Apply Y-axis mirroring for mirrored moves
        if is_mirrored:
            pos_offset = pos_offset.copy()  # Don't modify original
            ori_offset = ori_offset.copy()
            pos_offset[1] = -pos_offset[1]  # Mirror Y position
            ori_offset[2] = -ori_offset[2]  # Mirror yaw (rotation around Z)

        return MovementIntent(
            position=self.config.neutral_pos + pos_offset,
            orientation=self.config.neutral_eul + ori_offset,
            antennas=ant_offset,
        )

    def _subtract_noise(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        noise_profile: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Subtract noise profile from audio."""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        cleaned_magnitude = magnitude - (
            noise_profile[:, np.newaxis] * self.config.noise_subtraction_strength
        )
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0.0)

        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=512, length=len(audio))

        return cleaned_audio.astype(np.float32)

    def _clamp_bpm(self, bpm: float) -> float:
        """Force BPM into range by halving/doubling."""
        if bpm <= 0:
            return bpm
        while bpm < self.config.bpm_min:
            bpm *= 2.0
        while bpm > self.config.bpm_max:
            bpm /= 2.0
        return bpm

    def _audio_loop(self) -> None:
        """Audio analysis thread using SDK media manager."""
        try:
            self.mini.media.start_recording()
            logger.info(f"[{self.MODE_NAME}] Audio recording started")
            # Debug: Show device info
            logger.debug(
                f"[{self.MODE_NAME}] Audio backend: {type(self.mini.media).__name__}"
            )
            try:
                logger.debug(
                    f"[{self.MODE_NAME}] Input device: samplerate={self.mini.media.get_input_audio_samplerate()}Hz"
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] ERROR: Failed to start recording: {e}")
            return

        buf: np.ndarray[Any, np.dtype[np.float32]] = np.empty(0, dtype=np.float32)
        bpm_hist: collections.deque[float] = collections.deque(
            maxlen=self.config.bpm_stability_buffer
        )
        samples_received = 0

        try:
            while not self.stop_event.is_set():
                # Poll for audio samples from SDK
                try:
                    sample_raw: Optional[np.ndarray[Any, Any]] = (
                        self.mini.media.get_audio_sample()
                    )  # pyright: ignore[reportAssignmentType]
                except Exception:
                    break

                if sample_raw is None:
                    time.sleep(0.01)
                    continue

                # SDK returns stereo float32 with shape (n, 2)
                sample = cast(np.ndarray[Any, np.dtype[np.float32]], sample_raw)
                if sample.ndim == 2 and sample.shape[1] >= 2:
                    chunk = cast(
                        np.ndarray[Any, np.dtype[np.float32]],
                        (sample[:, 0] + sample[:, 1]) / 2.0,
                    )
                elif sample.ndim == 2 and sample.shape[1] == 1:
                    chunk = sample[:, 0]
                elif sample.ndim == 1:
                    chunk = sample
                else:
                    continue

                buf = np.append(buf, chunk).astype(np.float32)

                # Wait until we have a full analysis window
                if len(buf) < self.config.audio_buffer_len:
                    continue

                # We have enough data! Take the oldest window for analysis
                analysis_buf = buf[: self.config.audio_buffer_len]

                # Select noise profile based on current state
                with self.music_state.lock:
                    is_breathing = self.music_state.is_breathing

                # Apply noise subtraction
                if is_breathing and self.breathing_noise_profile is not None:
                    proc_buf = self._subtract_noise(
                        analysis_buf, self.breathing_noise_profile
                    )
                elif not is_breathing and self.dance_noise_profile is not None:
                    proc_buf = self._subtract_noise(
                        analysis_buf, self.dance_noise_profile
                    )
                else:
                    proc_buf = analysis_buf

                # Volume gate on cleaned audio
                rms = np.sqrt(np.mean(analysis_buf**2))
                cleaned_rms = np.sqrt(np.mean(proc_buf**2))

                if cleaned_rms < self.config.volume_gate_threshold:
                    with self.music_state.lock:
                        self.music_state.state = "Gathering"
                        self.music_state.librosa_bpm = 0.0
                        self.music_state.raw_amplitude = rms
                        self.music_state.last_event_time = 0.0
                        self.music_state.music_confident = False
                    # Slide window by chunk size and continue
                    buf = buf[len(chunk) :].astype(np.float32)
                    continue

                # Volume passed - run BPM detection
                confidence_threshold = (
                    self.config.volume_gate_threshold
                    * self.config.music_confidence_ratio
                )
                is_confident = cleaned_rms > confidence_threshold

                # BPM detection
                try:
                    tempo, beat_frames = librosa.beat.beat_track(
                        y=proc_buf,
                        sr=self.config.audio_rate,
                        units="frames",
                        tightness=80,
                    )
                except Exception as e:
                    logger.error(f"[{self.MODE_NAME}] ERROR in BPM detection: {e}")
                    buf = buf[len(chunk) :].astype(np.float32)
                    continue

                now = time.time()
                raw_tempo = float(
                    tempo[0]
                    if isinstance(tempo, np.ndarray) and tempo.size > 0
                    else tempo
                )
                tempo_val = self._clamp_bpm(raw_tempo)

                has_audio = (
                    cleaned_rms > self.config.volume_gate_threshold
                    and len(beat_frames) > 0
                )

                with self.music_state.lock:
                    if has_audio:
                        self.music_state.last_event_time = now
                    self.music_state.raw_librosa_bpm = raw_tempo
                    self.music_state.raw_amplitude = rms
                    self.music_state.cleaned_amplitude = np.abs(proc_buf).mean()
                    self.music_state.music_confident = is_confident

                    if tempo_val > 40:
                        bpm_hist.append(tempo_val)
                        self.music_state.librosa_bpm = float(np.mean(bpm_hist))

                    self.music_state.bpm_std = (
                        float(np.std(bpm_hist)) if len(bpm_hist) > 1 else 0.0
                    )

                    if len(bpm_hist) < self.config.bpm_stability_buffer:
                        self.music_state.state = "Gathering"
                    elif self.music_state.bpm_std < self.config.bpm_stability_threshold:
                        self.music_state.state = "Locked"
                        self.music_state.has_ever_locked = True
                    else:
                        self.music_state.state = "Unstable"

                # Slide window by one chunk
                buf = buf[len(chunk) :].astype(np.float32)

        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Error in audio loop: {e}")

        finally:
            # Stop recording
            logger.info(
                f"[{self.MODE_NAME}] Audio thread stopping, samples received: {samples_received}"
            )
            try:
                self.mini.media.stop_recording()
                logger.info(f"[{self.MODE_NAME}] Audio recording stopped")
            except Exception:
                # Common during shutdown if media manager closed first
                pass

    def _control_loop(self) -> None:
        """Movement control thread."""
        last_time = time.time()
        t_beats = 0.0
        breathing_time = 0.0
        is_executing_move = False
        move_beats_elapsed = 0.0
        force_breathing_until = 0.0
        last_active_bpm = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            with self.music_state.lock:
                librosa_bpm = self.music_state.librosa_bpm
                state = self.music_state.state
                last_event_time = self.music_state.last_event_time
                has_ever_locked = self.music_state.has_ever_locked
                music_confident = self.music_state.music_confident

            active_bpm = (
                librosa_bpm if now - last_event_time < self.config.silence_tmo else 0.0
            )

            # Separate criteria for STARTING vs CONTINUING a move:
            # START: Must be Locked AND music_confident (strict) - prevents ego-noise dancing
            # CONTINUE: Can coast through Unstable using last_active_bpm (loose)
            can_start_new_move = (
                active_bpm > 0
                and has_ever_locked
                and state == "Locked"
                and music_confident
            )

            # Update breathing state for audio thread
            with self.music_state.lock:
                self.music_state.is_breathing = not (
                    is_executing_move or can_start_new_move
                )

            in_forced_breathing = now < force_breathing_until

            if is_executing_move:
                # Continue executing move
                bpm_for_move = active_bpm if active_bpm > 0 else last_active_bpm
                beats_this_frame = dt * (bpm_for_move / 60.0)
                move_beats_elapsed += beats_this_frame
                t_beats += beats_this_frame

                if move_beats_elapsed >= self.config.beats_per_sequence:
                    # Move complete
                    is_executing_move = False
                    force_breathing_until = (
                        now + self.config.min_breathing_between_moves
                    )
                    self.choreographer.advance_move()
                else:
                    # Execute move frame
                    intent = self._compute_dance_pose(
                        t_beats, self.choreographer.current_move_name(), bpm_for_move
                    )
                    self.mixer.send_intent(intent)

            elif can_start_new_move and not in_forced_breathing:
                # Start new move
                is_executing_move = True
                t_beats = 0.0
                move_beats_elapsed = 0.0
                last_active_bpm = active_bpm
                logger.info(
                    f"[{self.MODE_NAME}] Starting: {self.choreographer.current_move_name()}"
                )

            else:
                # Breathing
                breathing_time += dt
                intent = self._compute_breathing_pose(breathing_time)
                self.mixer.send_intent(intent)

            time.sleep(0.01)
