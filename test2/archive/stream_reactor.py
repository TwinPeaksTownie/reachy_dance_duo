"""
Mode C: Stream Reactor - Real-time audio-to-motion dance mode.

Listens to system audio via loopback (BlackHole on Mac) and maps
frequency bands directly to robot movement in real-time.

This is the "Reactor" - visceral, arcade-like, sub-50ms latency.

Movement mapping:
- Bass (kick drums) → body_yaw (hip sway) + Z bounce
- Highs (hi-hats) → head pitch
- Vocals (filtered) → antenna spread
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DanceMode
from ..core.audio_stream import AudioStream
from ..core.safety_mixer import MovementIntent
from ..config import MODE_C_CONFIG, AUDIO_CONFIG
from .. import mode_settings

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


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
        bass_range = MODE_C_CONFIG["bass_range"]
        vocal_range = MODE_C_CONFIG["vocal_range"]
        high_range = MODE_C_CONFIG["high_range"]

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
        self.energy_history: deque = deque(
            maxlen=int(sample_rate / chunk_size * 0.5)
        )
        self.frames_since_beat = 0

    def process(self, audio_buffer: np.ndarray) -> dict[str, Any]:
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
        if norm_bass > 0.4 and raw_bass > (avg_energy * 1.5):
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


class StreamReactor(DanceMode):
    """Mode C: Real-time stream reactor.

    Maps audio frequencies to robot movement with minimal latency.
    """

    MODE_ID = "C"
    MODE_NAME = "Stream Reactor"

    def __init__(self, safety_mixer: SafetyMixer):
        super().__init__(safety_mixer)

        self.audio_stream: AudioStream | None = None
        self.feature_extractor: AudioFeatureExtractor | None = None

        # Dance state
        self.beat_counter = 0
        self.groove_intensity = 0.0
        self.last_beat_time = time.time()

        # Current smoothed positions (asymmetric attack/decay)
        self.curr = {
            "yaw": 0.0,
            "pitch": 0.0,
            "z": 0.0,
            "ant_l": -0.15,
            "ant_r": 0.15,
        }

        # Movement limits - load from mode_settings with fallback to config
        self._load_settings()

        # Physics (asymmetric attack/decay)
        self.PHYSICS = MODE_C_CONFIG["physics"]

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
        settings = mode_settings.get_mode_settings("C")
        self.MAX_YAW = settings.get("max_yaw", MODE_C_CONFIG["max_yaw"])
        self.MAX_PITCH = settings.get("max_pitch", MODE_C_CONFIG["max_pitch"])
        self.MAX_Z = settings.get("max_z", MODE_C_CONFIG["max_z"])

    def apply_settings(self, updates: dict[str, float]) -> None:
        """Apply settings updates (called from API for live tuning)."""
        if "max_yaw" in updates:
            self.MAX_YAW = updates["max_yaw"]
        if "max_pitch" in updates:
            self.MAX_PITCH = updates["max_pitch"]
        if "max_z" in updates:
            self.MAX_Z = updates["max_z"]

    async def start(self) -> None:
        """Start the stream reactor."""
        if self.running:
            return

        # Initialize audio
        self.audio_stream = AudioStream.create_for_mode("C", AUDIO_CONFIG["sample_rate"])
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=AUDIO_CONFIG["sample_rate"],
            chunk_size=AUDIO_CONFIG["chunk_size"],
        )

        # Reset dance state
        self.beat_counter = 0
        self.groove_intensity = 0.0
        self.last_beat_time = time.time()
        self.curr = {
            "yaw": 0.0,
            "pitch": 0.0,
            "z": 0.0,
            "ant_l": -0.15,
            "ant_r": 0.15,
        }

        # Start audio stream
        success = self.audio_stream.start(self._audio_callback)
        if not success:
            self._status["state"] = "error"
            return

        self.running = True
        self._status["running"] = True
        self._status["state"] = "dancing"
        print(f"[{self.MODE_NAME}] Started")

    async def stop(self) -> None:
        """Stop the stream reactor."""
        if not self.running:
            return

        self.running = False

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream = None

        # Return to neutral
        self.mixer.reset()

        self._status["running"] = False
        self._status["state"] = "idle"
        print(f"[{self.MODE_NAME}] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        # Convert any numpy types to Python native types for JSON serialization
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, 'item'):  # numpy scalar
                status[key] = value.item()
        return status

    def _audio_callback(self, samples: np.ndarray) -> None:
        """Process incoming audio and generate movement."""
        if not self.running or self.feature_extractor is None:
            return

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
            self.last_beat_time = time.time()
            self._status["beat_count"] = self.beat_counter
        else:
            self.groove_intensity = max(0.0, self.groove_intensity * 0.98)

        # Threshold bass to avoid drift on background noise
        clean_bass = 0.0 if features["bass"] < 0.15 else features["bass"]

        # Calculate targets
        # Body yaw: 4-beat pattern (left, center, right, center)
        directions = [1.0, 0.0, -1.0, 0.0]
        direction = directions[self.beat_counter]

        if time.time() - self.last_beat_time > 2.0:
            target_yaw = 0.0
        else:
            target_yaw = self.MAX_YAW * direction * clean_bass * self.groove_intensity

        # Head pitch: driven by highs
        target_pitch = -0.05 - (self.MAX_PITCH * features["high"])

        # Z bounce: duck down on bass
        target_z = 0.015 - (self.MAX_Z * clean_bass)

        # Antennas: driven by vocals (with snare subtraction)
        snare_penalty = features["high"] * 0.8
        clean_vocal = max(0.0, features["vocals"] - snare_penalty)
        vocal_drive = clean_vocal ** 2
        spread = 0.15 + (vocal_drive * 0.6)
        target_ant_l = -spread
        target_ant_r = spread

        # Apply asymmetric physics smoothing
        self._smooth("yaw", target_yaw, self.PHYSICS["body"])
        self._smooth("pitch", target_pitch, self.PHYSICS["head"])
        self._smooth("z", target_z, self.PHYSICS["z"])
        self._smooth("ant_l", target_ant_l, self.PHYSICS["ant"])
        self._smooth("ant_r", target_ant_r, self.PHYSICS["ant"])

        # Create movement intent and send to SafetyMixer
        intent = MovementIntent(
            position=np.array([0.0, 0.0, self.curr["z"]]),
            orientation=np.array([0.0, self.curr["pitch"], 0.0]),  # No head yaw
            antennas=np.array([self.curr["ant_l"], self.curr["ant_r"]]),
            body_yaw=self.curr["yaw"],
        )

        self.mixer.send_intent(intent)

    def _smooth(self, key: str, target: float, physics: dict) -> None:
        """Apply asymmetric smoothing (fast attack, slow decay)."""
        current = self.curr[key]

        # Determine if attacking or decaying
        is_attack = abs(target) > abs(current)
        alpha = physics["attack"] if is_attack else physics["decay"]

        self.curr[key] += (target - current) * alpha
