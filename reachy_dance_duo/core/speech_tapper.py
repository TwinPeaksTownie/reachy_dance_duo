"""SwayRollRT - Syllable-driven head nodding via onset detection.

Converts audio chunks into per-hop (10ms) motion offsets:
- Detects syllable onsets via energy derivative peaks
- Triggers discrete nod events (snap down, spring back)
- Voice activity detection with hysteresis (attack/release ramps)

The key insight: real nodding happens on syllable boundaries, not as
continuous oscillation. We detect energy "attacks" and trigger nods.
"""

from __future__ import annotations
import math
from collections import deque
from itertools import islice
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy import signal  # pyright: ignore[reportMissingImports]


# Tunables
SR = 16_000
FRAME_MS = 20
HOP_MS = 10

# VAD thresholds
VAD_DB_ON = -20.0  # Hardened threshold to filter BT connection noise & Music
VAD_DB_OFF = -45.0
VAD_ATTACK_MS = 150  # Extended attack to require sustained vocal energy
VAD_RELEASE_MS = 250

# Onset detection
ONSET_THRESHOLD = 5.0  # dB rise per hop to trigger onset
ONSET_COOLDOWN_MS = 150  # Minimum time between onsets (prevents double-triggers)
ONSET_LOOKBACK = 3  # Hops to look back for derivative

# Nod dynamics
NOD_PITCH_DEG = 15.0  # Increased for BOLD nod
NOD_ATTACK_MS = 40  # Time to snap forward
NOD_DECAY_MS = 10  # Instant snap back (1 frame)
NOD_Z_DROP_MM = 0.0  # Disabled to prevent IK collisions

# Face stabilization: compensate X to keep face position stable during nods
# When head pitches down, face moves forward; we counter with backward X translation
# This is the approximate distance from neck pivot to face center in mm
NECK_TO_FACE_MM = 0.0  # Disabled X-compensation to prevent IK collisions

# Subtle ambient sway (very reduced, not competing with nods)
SWAY_ROLL_DEG = 1.5  # Subtle roll
SWAY_ROLL_FREQ = 0.4  # Hz
SWAY_YAW_DEG = 2.0  # Subtle yaw drift
SWAY_YAW_FREQ = 0.15  # Hz

# Derived
FRAME = int(SR * FRAME_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
ATTACK_FR = max(1, int(VAD_ATTACK_MS / HOP_MS))
RELEASE_FR = max(1, int(VAD_RELEASE_MS / HOP_MS))
ONSET_COOLDOWN_FR = max(1, int(ONSET_COOLDOWN_MS / HOP_MS))
NOD_ATTACK_FR = max(1, int(NOD_ATTACK_MS / HOP_MS))
NOD_DECAY_FR = max(1, int(NOD_DECAY_MS / HOP_MS))


def _rms_dbfs(x: NDArray[np.float32]) -> float:
    """Root-mean-square in dBFS for float32 mono array in [-1,1]."""
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(x * x, dtype=np.float32) + 1e-12, dtype=np.float32)
    return float(20.0 * math.log10(float(rms) + 1e-12))


def _to_float32_mono(x: NDArray[Any]) -> NDArray[np.float32]:
    """Convert arbitrary PCM array to float32 mono in [-1,1].

    Accepts shapes: (N,), (1,N), (N,1), (C,N), (N,C).
    """
    a = np.asarray(x)
    if a.ndim == 0:
        return np.zeros(0, dtype=np.float32)

    # If 2D, decide which axis is channels (prefer small first dim)
    if a.ndim == 2:
        if a.shape[0] <= 8 and a.shape[0] <= a.shape[1]:
            a = np.mean(a, axis=0)
        else:
            a = np.mean(a, axis=1)
    elif a.ndim > 2:
        a = np.mean(a.reshape(a.shape[0], -1), axis=0)

    # Now 1D, cast/scale
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    # integer PCM
    info = np.iinfo(a.dtype)
    scale = float(max(-info.min, info.max))
    return a.astype(np.float32) / (scale if scale != 0.0 else 1.0)


def _resample_linear(
    x: NDArray[np.float32], sr_in: int, sr_out: int
) -> NDArray[np.float32]:
    """Lightweight linear resampler for short buffers."""
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32, endpoint=True)
    t_out = np.linspace(0.0, 1.0, num=n_out, dtype=np.float32, endpoint=True)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


class SwayRollRT:
    """Syllable-driven head nodding via onset detection.

    Usage:
        rt = SwayRollRT()
        results = rt.feed(pcm_int16_or_float, sr)  # List[dict]

    Each dict contains: pitch_rad, yaw_rad, roll_rad, x_mm, y_mm, z_mm
    (plus _deg variants for convenience).

    Key behavior:
    - Detects syllable onsets via energy derivative
    - Triggers discrete nod events (snap down, spring back)
    - Much more natural than continuous sine oscillation
    """

    def __init__(self, rng_seed: int = 7):
        """Initialize state."""
        self._seed = int(rng_seed)
        self.samples: deque[float] = deque(maxlen=10 * SR)
        self.carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        # VAD state
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0

        # Energy history for onset detection
        self._db_history: deque[float] = deque(maxlen=ONSET_LOOKBACK + 1)
        self._onset_cooldown = 0  # Frames until next onset allowed

        # Nod envelope state
        self._nod_phase = 0  # 0=idle, 1=attack, 2=decay
        self._nod_progress = 0  # Frames into current phase
        self._nod_amplitude = 0.0  # Current nod envelope [0, 1]

        # Ambient sway phases (randomized start)
        rng = np.random.default_rng(self._seed)
        self._phase_roll = float(rng.random() * 2 * math.pi)
        self._phase_yaw = float(rng.random() * 2 * math.pi)
        self.t = 0.0

        # Stats
        self._onset_count = 0

        # Vocal Bandpass Filter (200Hz - 3000Hz)
        # Digital Earmuffs: Block Low Rumble & High Guitars/Cymbals
        # Standard Telephony Band is ~300Hz-3400Hz. We use 200-3000Hz.
        nyquist = SR / 2
        low = 200.0 / nyquist
        high = 3000.0 / nyquist
        self._sos = signal.butter(4, [low, high], btype="band", output="sos")
        # Initialize filter state (zeros for silence start)
        self._zi = signal.sosfilt_zi(self._sos)

    def reset(self) -> None:
        """Reset state (VAD/env/buffers/time) but keep initial phases/seed."""
        self.samples.clear()
        self.carry = np.zeros(0, dtype=np.float32)
        self.vad_on = False
        self.vad_above = 0
        self.vad_below = 0
        self._db_history.clear()
        self._onset_cooldown = 0
        self._nod_phase = 0
        self._nod_progress = 0
        self._nod_amplitude = 0.0
        self.t = 0.0
        self._onset_count = 0

    def feed(self, pcm: NDArray[Any], sr: int | None) -> List[Dict[str, float]]:
        """Stream in PCM chunk. Returns a list of sway dicts, one per hop (HOP_MS).

        Args:
            pcm: np.ndarray, shape (N,) or (C,N)/(N,C); int or float.
            sr:  sample rate of `pcm` (None -> assume SR).

        """
        sr_in = SR if sr is None else int(sr)
        x = _to_float32_mono(pcm)
        if x.size == 0:
            return []
        if sr_in != SR:
            x = _resample_linear(x, sr_in, SR)
            if x.size == 0:
                return []

        # Apply Bandpass Filter (Telephony Band: 200Hz-3000Hz)
        # Keeps RMS/VAD focused on Voice, ignores Guitars/Drums.
        if x.size > 0:
            x, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
            x = x.astype(np.float32, copy=False)

        # append to carry and consume fixed HOP chunks
        if self.carry.size:
            self.carry = np.concatenate([self.carry, x])
        else:
            self.carry = x

        out: List[Dict[str, float]] = []

        while self.carry.size >= HOP:
            hop = self.carry[:HOP]
            remaining: NDArray[np.float32] = self.carry[HOP:]
            self.carry = remaining

            self.samples.extend(hop.tolist())
            if len(self.samples) < FRAME:
                self.t += HOP_MS / 1000.0
                continue

            frame = np.fromiter(
                islice(self.samples, len(self.samples) - FRAME, len(self.samples)),
                dtype=np.float32,
                count=FRAME,
            )
            db = _rms_dbfs(frame)

            # VAD with hysteresis + attack/release
            if db >= VAD_DB_ON:
                self.vad_above += 1
                self.vad_below = 0
                if not self.vad_on and self.vad_above >= ATTACK_FR:
                    self.vad_on = True
            elif db <= VAD_DB_OFF:
                self.vad_below += 1
                self.vad_above = 0
                if self.vad_on and self.vad_below >= RELEASE_FR:
                    self.vad_on = False

            # Store dB for onset detection
            self._db_history.append(db)

            # Decrement onset cooldown
            if self._onset_cooldown > 0:
                self._onset_cooldown -= 1

            # Onset detection: derivative spike while VAD is on
            onset_detected = False
            if (
                self.vad_on
                and len(self._db_history) >= ONSET_LOOKBACK + 1
                and self._onset_cooldown == 0
            ):
                # Compute derivative (dB rise over lookback window)
                db_old = self._db_history[0]
                db_new = self._db_history[-1]
                db_rise = db_new - db_old

                if db_rise >= ONSET_THRESHOLD:
                    onset_detected = True
                    self._onset_cooldown = ONSET_COOLDOWN_FR
                    self._onset_count += 1
                    # Trigger nod: start attack phase
                    self._nod_phase = 1
                    self._nod_progress = 0

            # Update nod envelope
            if self._nod_phase == 1:
                # Attack: ramp up
                self._nod_progress += 1
                self._nod_amplitude = min(1.0, self._nod_progress / NOD_ATTACK_FR)
                if self._nod_progress >= NOD_ATTACK_FR:
                    # Transition to decay
                    self._nod_phase = 2
                    self._nod_progress = 0
            elif self._nod_phase == 2:
                # Decay: ramp down
                self._nod_progress += 1
                self._nod_amplitude = max(0.0, 1.0 - self._nod_progress / NOD_DECAY_FR)
                if self._nod_progress >= NOD_DECAY_FR:
                    # Return to idle
                    self._nod_phase = 0
                    self._nod_progress = 0
                    self._nod_amplitude = 0.0

            self.t += HOP_MS / 1000.0

            # Compute motion
            # Nod: pitch goes POSITIVE (look forward/down) during nod
            nod_pitch = math.radians(NOD_PITCH_DEG) * self._nod_amplitude
            nod_z = -NOD_Z_DROP_MM * self._nod_amplitude

            # Face stabilization: compensate X to keep face visually stationary
            # When pitch is negative (tilting down), face moves forward in world frame
            # We counter by moving backward (negative X in local frame)
            # For a rotation θ about Y-axis, face offset is d*sin(θ) forward
            # nod_pitch is negative for down-nod, so sin(nod_pitch) is negative
            # We want to move backward (negative X), so: x = d * sin(pitch)
            # (negative pitch → negative sin → negative X ✓)
            nod_x = NECK_TO_FACE_MM * math.sin(nod_pitch)

            # Subtle ambient sway (only when VAD is on, very subdued)
            vad_env = 1.0 if self.vad_on else 0.0
            ambient_roll = (
                math.radians(SWAY_ROLL_DEG)
                * vad_env
                * 0.3  # Further reduced
                * math.sin(2 * math.pi * SWAY_ROLL_FREQ * self.t + self._phase_roll)
            )
            ambient_yaw = (
                math.radians(SWAY_YAW_DEG)
                * vad_env
                * 0.3  # Further reduced
                * math.sin(2 * math.pi * SWAY_YAW_FREQ * self.t + self._phase_yaw)
            )

            out.append(
                {
                    "pitch_rad": nod_pitch,
                    "yaw_rad": ambient_yaw,
                    "roll_rad": ambient_roll,
                    "pitch_deg": math.degrees(nod_pitch),
                    "yaw_deg": math.degrees(ambient_yaw),
                    "roll_deg": math.degrees(ambient_roll),
                    "x_mm": nod_x,
                    "y_mm": 0.0,
                    "z_mm": nod_z,
                    # Additional metadata for debugging/visualization
                    "_onset": onset_detected,
                    "_nod_amplitude": self._nod_amplitude,
                    # Signal for MotionController to freeze face tracking
                    # Use Phase check (1=Attack, 2=Decay) to cover entire event
                    "is_nodding": self._nod_phase != 0,
                },
            )

        return out
