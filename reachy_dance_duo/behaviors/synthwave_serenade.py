"""Synthwave Serenade (Lite) - Real-time audio-to-motion dance mode.

Listens to system audio via loopback (BlackHole on Mac, PulseAudio on Linux) and maps
frequency bands to robot movement in real-time using lightweight FFT analysis.

Architecture:
- MotionController handles: breathing, face tracking, body anchor, final robot commands
- SynthwaveSerenade computes: audio-reactive motion offsets (FFT-based)
- Motion flows through add_dance_offset()
This is the "Reactor" - visceral, arcade-like, sub-50ms latency.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .. import mode_settings
from ..config import AUDIO_CONFIG, MODE_C_CONFIG
from ..core.audio_stream import AudioStream
from .base import DanceMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from ..core.safety_mixer import SafetyMixer
    from ..core.motion_controller import MotionController


# ============================================================================
# Audio Feature Extractor (from Stream Reactor)
# ============================================================================


class AudioFeatureExtractor:
    """Extract audio features from FFT spectrum.

    Divides spectrum into bands and detects beats.
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # FFT frequency bins
        self.freqs = np.fft.rfftfreq(chunk_size, 1 / sample_rate)

        # Frequency band indices
        bass_range = cast(tuple[int, int], MODE_C_CONFIG["bass_range"])
        vocal_range = cast(tuple[int, int], MODE_C_CONFIG["vocal_range"])
        high_range = cast(tuple[int, int | None], MODE_C_CONFIG["high_range"])

        self.idx_bass = np.where(
            (self.freqs >= bass_range[0]) & (self.freqs <= bass_range[1])
        )[0]
        self.idx_vocal = np.where(
            (self.freqs >= vocal_range[0]) & (self.freqs <= vocal_range[1])
        )[0]
        self.idx_high = np.where(self.freqs >= high_range[0])[0]

        # Adaptive normalization state
        self.max_bass = 0.01
        self.max_vocal = 0.01
        self.max_high = 0.01

        # Transient suppression for vocal isolation
        self.prev_vocal_energy = 0.0

        # Beat detection
        self.energy_history: deque[float] = deque(
            maxlen=int(sample_rate / chunk_size * 0.5)
        )
        self.frames_since_beat = 0

    def process(
        self, audio_buffer: np.ndarray[Any, np.dtype[np.float32]]
    ) -> dict[str, Any]:
        """Process audio buffer and extract features.

        Returns:
            Dictionary with:
            - is_beat: bool
            - bass: float (0-1)
            - vocals: float (0-1)
            - high: float (0-1)
        """
        # Apply window and compute FFT
        windowed = audio_buffer * np.hanning(len(audio_buffer))
        fft_spec = np.abs(np.fft.rfft(windowed))

        # Extract raw band energies
        raw_bass = np.mean(fft_spec[self.idx_bass]) if len(self.idx_bass) else 0
        raw_vocal = np.mean(fft_spec[self.idx_vocal]) if len(self.idx_vocal) else 0
        raw_high = np.mean(fft_spec[self.idx_high]) if len(self.idx_high) else 0

        # Vocal isolation via slew-rate limiting
        # Snares spike instantly, vocals ramp up gradually
        rise_limit = self.max_vocal * 0.1
        if raw_vocal > self.prev_vocal_energy + rise_limit:
            filtered_vocal = self.prev_vocal_energy + rise_limit
        else:
            filtered_vocal = raw_vocal
        self.prev_vocal_energy = filtered_vocal

        # Adaptive normalization with decay
        decay = 0.95
        self.max_bass = max(raw_bass, self.max_bass * decay)
        self.max_vocal = max(filtered_vocal, self.max_vocal * decay)
        self.max_high = max(raw_high, self.max_high * decay)

        # Normalize and square for contrast
        norm_bass = (raw_bass / (self.max_bass + 1e-6)) ** 2
        norm_vocal = (filtered_vocal / (self.max_vocal + 1e-6)) ** 2
        norm_high = (raw_high / (self.max_high + 1e-6)) ** 2

        # Beat detection
        is_beat = False
        avg_energy = np.mean(self.energy_history) if self.energy_history else 0

        # Noise gate: only detect beats if raw bass energy is significant
        NOISE_THRESHOLD = 0.5

        if (
            raw_bass > NOISE_THRESHOLD
            and norm_bass > 0.4
            and raw_bass > (avg_energy * 1.5)
        ):
            min_frames = int(self.sample_rate / self.chunk_size * 0.25)
            if self.frames_since_beat > min_frames:
                is_beat = True
                self.frames_since_beat = 0

        self.frames_since_beat += 1
        self.energy_history.append(raw_bass)

        return {
            "is_beat": is_beat,
            "bass": float(np.clip(norm_bass, 0, 1)),
            "vocals": float(np.clip(norm_vocal, 0, 1)),
            "high": float(np.clip(norm_high, 0, 1)),
        }


# ============================================================================
# Synthwave Serenade (Lite)
# ============================================================================


class SynthwaveSerenade(DanceMode):
    """Synthwave Serenade - Real-time computer audio streaming mode.

    Maps audio frequencies to robot movement with minimal latency.
    Uses BlackHole on macOS, PulseAudio on Linux.
    """

    MODE_ID = "synthwave_serenade"
    MODE_NAME = "Synthwave Serenade (Lite)"

    def __init__(
        self,
        safety_mixer: SafetyMixer,
        mini: ReachyMini,
        motion_controller: MotionController,
    ):
        """Initialize the Synthwave Serenade."""
        # Note: DanceMode base class expects safety_mixer, but we use motion_controller
        self.motion_controller = motion_controller
        self.running = False

        self.audio_stream: AudioStream | None = None
        self.feature_extractor: AudioFeatureExtractor | None = None

        # Time tracking for oscillators
        self._start_time: float = 0.0
        self._t: float = 0.0

        # Dance state
        self.beat_counter = 0
        self.groove_intensity = 0.0
        self.last_beat_time = time.time()

        # Smoothed audio features (for oscillator modulation)
        self._smoothed_bass: float = 0.0
        self._smoothed_high: float = 0.0
        self._smoothed_vocal: float = 0.0

        # Current head roll (beat-driven tilt)
        self._head_roll: float = 0.0

        # Antenna beat flick (snaps to 0 on beat, decays fast)
        self._antenna_flick: float = 0.0

        # Load settings
        self._load_settings()

        # Physics (asymmetric attack/decay for smoothing audio features)
        self.PHYSICS = {
            "bass": {"attack": 0.4, "decay": 0.1},
            "high": {"attack": 0.5, "decay": 0.15},
            "vocal": {"attack": 0.3, "decay": 0.1},
            "roll": {"attack": 0.25, "decay": 0.15},
        }

        # Status tracking
        self._status = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "beat_count": 0,
            "bass": 0.0,
            "vocals": 0.0,
            "high": 0.0,
        }

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("synthwave_serenade")
        if not settings:
            settings = mode_settings.get_mode_settings("bluetooth_streamer")
        if not settings:
            settings = {}

        # Meta-parameters from UI
        self._intensity: float = float(settings.get("intensity") or 1.0)
        self._liveliness: float = float(settings.get("liveliness") or 1.0)
        self._strain_threshold: float = float(settings.get("strain_threshold") or 15.0)

        # Apply meta-parameters to underlying values
        self._apply_meta_params()

    def _apply_meta_params(self) -> None:
        """Apply intensity and liveliness to underlying parameters."""
        # Base values (at intensity=1.0)
        BASE_ROLL = 0.4
        BASE_PITCH = 0.3
        BASE_Z = -0.01

        # Intensity scales movement amplitude
        self.MAX_ROLL = BASE_ROLL * self._intensity
        self.MAX_PITCH = BASE_PITCH * self._intensity
        self.MAX_Z = BASE_Z

        # Base oscillator values (at liveliness=1.0)
        BASE_ANTENNA_AMP = 1.5
        BASE_PITCH_AMP = 0.15

        # Liveliness scales Antennas (Sway), Intensity scales everything
        self.ANTENNA_AMP = BASE_ANTENNA_AMP * self._liveliness * self._intensity
        self.PITCH_AMP = BASE_PITCH_AMP * self._intensity

        # Fixed values (not exposed to UI)
        self.ANTENNA_FREQ = 0.5  # Slow sway: 2 seconds per cycle
        self.PITCH_FREQ = 4.5

        # Update MotionController strain threshold if available
        if hasattr(self, "motion_controller") and self.motion_controller:
            self.motion_controller._strain_threshold = self._strain_threshold

    def apply_settings(self, updates: dict[str, float]) -> None:
        """Apply settings updates (called from API for live tuning)."""
        if "intensity" in updates:
            self._intensity = updates["intensity"]
        if "liveliness" in updates:
            self._liveliness = updates["liveliness"]
        if "strain_threshold" in updates:
            self._strain_threshold = updates["strain_threshold"]

        # Re-apply meta-parameters when any setting changes
        self._apply_meta_params()

    async def start(self) -> None:
        """Start the synthwave serenade dancer."""
        if self.running:
            return

        # Check if pyaudio is available
        from ..core.audio_stream import PYAUDIO_AVAILABLE

        if not PYAUDIO_AVAILABLE:
            logger.error("[Synthwave Serenade] ERROR: pyaudio is not installed.")
            self._status["state"] = "error"
            return

        # Start MotionController's motion loop (breathing, face tracking, etc.)
        self.motion_controller.start()

        # Initialize audio
        self.audio_stream = AudioStream.create_for_mode(
            "synthwave_serenade",
            sample_rate=AUDIO_CONFIG["sample_rate"],
            chunk_size=AUDIO_CONFIG["chunk_size"],
        )
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=AUDIO_CONFIG["sample_rate"],
            chunk_size=AUDIO_CONFIG["chunk_size"],
        )

        # Reset state
        self._start_time = time.time()
        self._t = 0.0
        self.beat_counter = 0
        self.groove_intensity = 0.0
        self.last_beat_time = time.time()
        self._smoothed_bass = 0.0
        self._smoothed_high = 0.0
        self._smoothed_vocal = 0.0
        self._head_roll = 0.0
        self._antenna_flick = 0.0

        # Note: Face tracking frames are fed by the main app loop

        # Start audio stream
        success = self.audio_stream.start(self._audio_callback)
        if not success:
            self._status["state"] = "error"
            return

        self.running = True
        self._status["running"] = True
        self._status["state"] = "dancing"
        logger.info("[Synthwave Serenade] Started (Lite Mode)")

    async def stop(self) -> None:
        """Stop the system audio dancer."""
        if not self.running:
            return

        logger.info(f"[Synthwave Serenade] {time.strftime('%H:%M:%S')} stop() called")
        self.running = False

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream = None

        # Clear offsets and stop motion controller
        self.motion_controller.clear_dance_offset()
        self.motion_controller.clear_vocal_offset()
        self.motion_controller.set_breathing_scale(1.0)
        self.motion_controller.stop()

        self._status["running"] = False
        self._status["state"] = "idle"
        logger.info("[Synthwave Serenade] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, "item"):
                status[key] = value.item()
        return status

    def _smooth_feature(
        self, current: float, target: float, physics: dict[str, float]
    ) -> float:
        """Apply asymmetric smoothing to audio feature."""
        is_attack = target > current
        alpha = physics["attack"] if is_attack else physics["decay"]
        return float(current + (target - current) * alpha)

    def _audio_callback(self, samples: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        """Process incoming audio, extract features, and update motion."""
        if not self.running or self.feature_extractor is None:
            return

        # Update time
        current_time = time.time()
        self._t = current_time - self._start_time

        # Extract features
        features = self.feature_extractor.process(samples)

        # Update status
        self._status["bass"] = features["bass"]
        self._status["vocals"] = features["vocals"]
        self._status["high"] = features["high"]

        # Process beat
        if features["is_beat"]:
            self.beat_counter = (self.beat_counter + 1) % 4
            self.groove_intensity = 1.0
            self.last_beat_time = current_time
            self._status["beat_count"] = self.beat_counter
        else:
            self.groove_intensity = max(0.0, self.groove_intensity * 0.98)

        # Smooth audio features
        clean_bass = 0.0 if features["bass"] < 0.15 else features["bass"]
        self._smoothed_bass = self._smooth_feature(
            self._smoothed_bass, clean_bass, self.PHYSICS["bass"]
        )
        self._smoothed_high = self._smooth_feature(
            self._smoothed_high, features["high"], self.PHYSICS["high"]
        )

        # Vocal with snare subtraction
        snare_penalty = features["high"] * 0.8
        clean_vocal = max(0.0, features["vocals"] - snare_penalty)
        self._smoothed_vocal = self._smooth_feature(
            self._smoothed_vocal, clean_vocal, self.PHYSICS["vocal"]
        )

        # Breathing scale: dampen when speech/vocals detected
        if self._smoothed_vocal > 0.3:
            self.motion_controller.set_breathing_scale(1.0)
        else:
            self.motion_controller.set_breathing_scale(1.0)

        # ================================================================
        # Movement Generation (Lite Logic)
        # ================================================================

        # Antenna: constant sine wave + beat flick + vocal spread
        # Base sine oscillation
        antenna_min = 0.05
        antenna_max = 0.8
        antenna_mid = (antenna_min + antenna_max) / 2
        antenna_range = (antenna_max - antenna_min) / 2

        sine_val = np.sin(2 * np.pi * self.ANTENNA_FREQ * self._t)
        antenna_base = antenna_mid + antenna_range * sine_val

        # Beat flick: snap toward 0 on beat
        if features["is_beat"]:
            self._antenna_flick = 1.0
        else:
            self._antenna_flick = max(0.0, self._antenna_flick - 0.15)

        # Vocal spread (from Stream Reactor logic: vocals -> spread)
        # StreamReactor: spread = 0.15 + (vocal_drive * 0.6)
        vocal_drive = self._smoothed_vocal**2
        vocal_spread = 0.15 + (vocal_drive * 0.6)

        # Combine base sway with flick and vocal spread
        # If vocals are strong, override sway with spread
        # Logic: blend between base sway and vocal spread based on vocal intensity?
        # Or just add them?
        # StreamReactor logic was: target_ant_l = -spread, target_ant_r = spread
        # Let's use the Synthwave logic but inject vocal influence

        # Apply flick (lerp toward 0 when flicking)
        antenna_val = antenna_base * (1.0 - self._antenna_flick * 0.9)

        # Add vocal spread influence (widen when singing)
        antenna_val += vocal_spread * 0.5

        ant_left = -antenna_val
        ant_right = antenna_val

        # Head pitch: Start at small down, snap forward on highs/beats
        pitch_pulse = (
            self.PITCH_AMP
            * self._smoothed_high
            * abs(np.sin(np.pi * self.PITCH_FREQ * self._t))
        )
        dance_pitch = (0.2 * self._intensity) + pitch_pulse

        # Base Z offset (duck on bass from StreamReactor logic: target_z = 0.015 - (MAX_Z * clean_bass))
        # Synthwave logic: BASE_Z = -0.01.
        # Feature: Bass -> Z bounce
        # We can modulate Z with bass
        z_bounce = self.MAX_Z * self._smoothed_bass
        dance_z = -0.01 + z_bounce

        # Head roll (beat-driven tilt)
        directions = [1.0, 0.0, -1.0, 0.0]
        direction = directions[self.beat_counter]

        if current_time - self.last_beat_time > 2.0:
            target_head_roll = 0.0
        else:
            target_head_roll = (
                self.MAX_ROLL * direction * self._smoothed_bass * self.groove_intensity
            )

        self._head_roll = self._smooth_feature(
            self._head_roll, target_head_roll, self.PHYSICS["roll"]
        )

        # Send to MotionController
        dance_position = np.array([0.0, 0.0, dance_z], dtype=np.float32)
        dance_orientation = np.array(
            [self._head_roll, dance_pitch, 0.0], dtype=np.float32
        )

        self.motion_controller.add_dance_offset(
            position=dance_position,
            orientation=dance_orientation,
            antennas=(float(ant_left), float(ant_right)),
            body_yaw=0.0,  # MotionController handles body anchor
        )


# Legacy aliases
SystemAudioDancer = SynthwaveSerenade
BluetoothStreamer = SynthwaveSerenade
DiscoAudioDancer = SynthwaveSerenade
