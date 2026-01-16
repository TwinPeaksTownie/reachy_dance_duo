"""
SafetyMixer - The central "spinal cord" of the dance system.

All movement commands flow through here. Modes generate intent,
SafetyMixer determines reality by applying:
1. Inverse collision limiting (low Z → force pitch up)
2. LERP-based smoothing to remove jitter
3. Absolute limits enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

from reachy_mini import utils


@dataclass
class MovementIntent:
    """Movement request from a dance mode.

    Modes generate these to express what they want. SafetyMixer
    determines what actually happens.
    """

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # [x, y, z] meters
    orientation: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # [roll, pitch, yaw] radians
    antennas: np.ndarray = field(
        default_factory=lambda: np.array([-0.15, 0.15])
    )  # [left, right]
    body_yaw: float = 0.0  # For Mode C hip sway (radians)

    def copy(self) -> MovementIntent:
        """Create a deep copy."""
        return MovementIntent(
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            antennas=self.antennas.copy(),
            body_yaw=self.body_yaw,
        )


@dataclass
class SafetyConfig:
    """Tunable safety parameters.

    These can be adjusted via the config UI in real-time.
    """

    # Inverse collision limiter
    # When Z drops below threshold, pitch gets clamped to prevent chin-chest collision
    z_threshold: float = 0.005  # meters - below this, pitch limiting kicks in
    max_pitch_at_low_z: float = (
        0.15  # radians (~8.6 deg) - max downward pitch when Z is low
    )

    # Smoothing (LERP factor)
    # 0 = no movement, 1 = instant snap, 0.1-0.2 = smooth organic motion
    smoothing_alpha: float = 0.15

    # Absolute position limits [x, y, z] in meters
    max_position: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.028])
    )
    min_position: np.ndarray = field(
        default_factory=lambda: np.array([-0.05, -0.05, -0.03])
    )

    # Absolute orientation limits [roll, pitch, yaw] in radians
    max_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.35, 0.7])
    )  # ~28, 20, 40 deg
    min_orientation: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.4, -0.7])
    )

    # Antenna limits (wider range for beat-reactive drops)
    # User requested 3.15 max (approx Pi). We set safety to 4.0 to allow full range.
    max_antenna: float = 4.0
    min_antenna: float = -4.0

    # Body yaw limits (for Mode C hip sway)
    max_body_yaw: float = 0.8  # ~46 degrees
    min_body_yaw: float = -0.8

    # Global intensity scalar (0.0 to 1.0) - dampens all movement
    intensity: float = 1.0


class SafetyMixer:
    """Central safety layer - all movement commands flow through here.

    Modes call send_intent() with their desired movement. SafetyMixer
    applies safety transforms and sends the result to the robot.
    """

    def __init__(self, config: SafetyConfig, mini: ReachyMini):
        self.config = config
        self.mini = mini

        # Current smoothed state (starts at neutral)
        self._current = MovementIntent()

        # Track if we've ever sent a command
        self._initialized = False

    def send_intent(self, intent: MovementIntent) -> None:
        """Apply safety transforms and send to robot.

        This is the main entry point. All modes should call this
        instead of directly calling mini.set_target().
        """
        # Apply intensity scaling first
        scaled = self._apply_intensity(intent)

        # Apply inverse collision limiter
        safe = self._apply_collision_limits(scaled)

        # Apply smoothing
        smoothed = self._apply_smoothing(safe)

        # Clamp to absolute limits
        clamped = self._clamp_to_limits(smoothed)

        # Send to robot
        self._send_to_robot(clamped)

    def _apply_intensity(self, intent: MovementIntent) -> MovementIntent:
        """Scale all movement by global intensity factor."""
        if self.config.intensity >= 1.0:
            return intent

        result = intent.copy()
        result.position = intent.position * self.config.intensity
        result.orientation = intent.orientation * self.config.intensity
        # Keep antennas at base spread, scale the delta
        base_antennas = np.array([-0.15, 0.15])
        antenna_delta = intent.antennas - base_antennas
        result.antennas = base_antennas + (antenna_delta * self.config.intensity)
        result.body_yaw = intent.body_yaw * self.config.intensity
        return result

    def _apply_collision_limits(self, intent: MovementIntent) -> MovementIntent:
        """Inverse limiter: as Z goes down, force pitch up.

        This prevents the chin from smashing into the chest when
        the head crouches down.
        """
        result = intent.copy()

        z = intent.position[2]
        pitch = intent.orientation[1]  # Index 1 = pitch

        if z < self.config.z_threshold:
            # Head is low - don't let it look down too far (chin collision)
            # Pitch is positive when looking down, so we clamp the maximum
            max_allowed_pitch = self.config.max_pitch_at_low_z
            result.orientation[1] = min(pitch, max_allowed_pitch)

        elif z > 0.025:
            # Head is high - joint extension is limited
            # Clamp downward pitch to prevent "Head pose not achievable"
            max_high_pitch = 0.2  # ~11 deg
            result.orientation[1] = min(pitch, max_high_pitch)

        return result

    def _apply_smoothing(self, intent: MovementIntent) -> MovementIntent:
        """LERP toward target to remove jitter.

        Uses exponential smoothing (low-pass filter) to create
        organic, flowing motion instead of jerky robot movements.

        NOTE: Antennas bypass smoothing for snappy beat-reactive drops.
        They can't collide with anything so safety smoothing isn't needed.
        """
        alpha = self.config.smoothing_alpha

        if not self._initialized:
            # First command - snap to position
            self._current = intent.copy()
            self._initialized = True
            return self._current.copy()

        # LERP head position/orientation (needs smoothing for safety)
        self._current.position = (
            self._current.position * (1 - alpha) + intent.position * alpha
        )
        self._current.orientation = (
            self._current.orientation * (1 - alpha) + intent.orientation * alpha
        )
        self._current.body_yaw = (
            self._current.body_yaw * (1 - alpha) + intent.body_yaw * alpha
        )

        # Antennas: NO smoothing - snap directly for beat-reactive drops
        self._current.antennas = intent.antennas.copy()

        return self._current.copy()

    def _clamp_to_limits(self, intent: MovementIntent) -> MovementIntent:
        """Enforce absolute hardware limits."""
        result = intent.copy()

        # Clamp position
        result.position = np.clip(
            intent.position,
            self.config.min_position,
            self.config.max_position,
        )

        # Clamp orientation
        result.orientation = np.clip(
            intent.orientation,
            self.config.min_orientation,
            self.config.max_orientation,
        )

        # Clamp antennas
        result.antennas = np.clip(
            intent.antennas,
            self.config.min_antenna,
            self.config.max_antenna,
        )

        # Clamp body yaw
        result.body_yaw = np.clip(
            intent.body_yaw,
            self.config.min_body_yaw,
            self.config.max_body_yaw,
        )

        return result

    def _send_to_robot(self, intent: MovementIntent) -> None:
        """Send the safe movement to the robot."""
        pose = utils.create_head_pose(
            *intent.position,
            *intent.orientation,
            degrees=False,
        )

        self.mini.set_target(
            pose,
            antennas=[float(x) for x in intent.antennas],
            body_yaw=intent.body_yaw,
        )

    def reset(self) -> None:
        """Reset to neutral position."""
        self._current = MovementIntent()
        self._initialized = False
        self._send_to_robot(self._current)

    def update_config(self, **kwargs) -> None:
        """Update config parameters (for live UI tuning)."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_current_state(self) -> MovementIntent:
        """Get the current smoothed state (for UI display)."""
        return self._current.copy()
