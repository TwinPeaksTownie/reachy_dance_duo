"""Motion Controller for Reachy Ultradancemix 9000.

Implements:
- Breathing motion (roll + antenna sway)
- Euler composition (breathing roll + pitch/yaw)
- Body anchor system (ANCHORED/SYNCING/STABILIZING)

Based on the reachy_live_vlm motion controller - the one that "feels alive".
"""

from __future__ import annotations

from enum import Enum
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from numpy.typing import NDArray  # pyright: ignore[reportMissingImports]
from scipy.spatial.transform import Rotation as R  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from reachy_mini import ReachyMini  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# ============================================================================
# Body Anchor State Machine
# ============================================================================


class AnchorState(Enum):
    """Body anchor states for smooth body following."""

    ANCHORED = "anchored"  # Body locked at anchor point
    SYNCING = "syncing"  # Body interpolating toward head
    STABILIZING = "stabilizing"  # Waiting for head to stabilize


def ease_in_out(t: float) -> float:
    """Ease-in-out interpolation curve."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


# ============================================================================
# Motion Controller
# ============================================================================


class MotionController:
    """Manages face tracking, breathing, and body anchor for Reachy Mini.

    This is the "spinal cord" that makes motion feel alive:
    1. Breathing layer (always running) - subtle oscillation
    2. Face tracking layer (when face detected) - look at people
    3. Body anchor system - body follows head with threshold

    Dance modes ADD their motion on top via `add_dance_offset()`.
    """

    def __init__(self, mini: ReachyMini, enabled: bool = True) -> None:
        """Initialize motion controller.

        Args:
            mini: ReachyMini SDK instance
            enabled: Whether to start with motion control enabled

        """
        self.mini = mini
        self.enabled = enabled
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Face offset (stored as OFFSET from neutral, not absolute pose)
        self._face_lock = threading.Lock()
        self._current_face_offset: Optional[NDArray] = None  # 4x4 offset matrix
        self._last_known_face_offset: Optional[NDArray] = None
        self._last_face_time: float = 0.0
        self._face_lost_timeout: float = 2.0
        self._face_lost: bool = False

        # Neutral pose system (Dynamic Reanchoring)
        # _home_pose: The fixed, wide-stance starting position
        self._home_pose = np.eye(4, dtype=np.float32)
        self._home_pose[0, 3] = -0.005  # x-offset
        self._home_pose[2, 3] = 0.01  # z-lift

        # _neutral_pose: The DYNAMIC anchor point that drifts towards the user
        self._neutral_pose = self._home_pose.copy()

        # Anchoring parameters
        self._anchor_drift_rate = (
            0.05  # How fast neutral drifts to target per frame (0.0-1.0)
        )
        self._return_to_home_rate = 0.01  # How fast we return to home when face lost

        # Face tracking smoothing (LPF to prevent fighting nods)
        self._face_pitch_smoothed: float = 0.0
        self._face_tracking_inhibited: bool = False
        self._inhibition_expiry: float = 0.0
        self._inhibition_cooldown: float = 0.2  # 200ms lock after nod ends

        # Pitch interpolation (kept for potential future use)
        self._current_pitch: float = 0.0
        self._pitch_target: Optional[float] = None
        self._pitch_interp_start: float = 0.0
        self._pitch_interp_start_time: float = 0.0
        self._pitch_interp_duration: float = 0.5
        self._pitch_hysteresis: float = np.deg2rad(1.0)

        # Rate limiting for pitch
        self._last_commanded_pitch: float = 0.0
        self._max_pitch_change_per_frame: float = np.deg2rad(5.0)

        # Body anchor state machine
        self._anchor_state = AnchorState.ANCHORED
        self._body_anchor_yaw: float = 0.0  # degrees
        self._strain_threshold: float = 13.0  # degrees
        self._sync_start_time: float = 0.0
        self._sync_duration: float = 1.0
        self._sync_start_yaw: float = 0.0
        self._sync_target_yaw: float = 0.0

        # Stability tracking for STABILIZING state
        self._stability_start_time: float = 0.0
        self._stability_duration: float = 3.0
        self._stability_threshold: float = 2.0
        self._last_head_yaw: float = 0.0

        # Breathing parameters (Lissajous: roll + y-sway)
        self._breathing_roll_amplitude: float = np.deg2rad(4.0)
        self._breathing_y_amplitude: float = 0.003  # 3mm y-sway
        self._breathing_frequency: float = 0.2  # Hz (5s cycle)
        self._antenna_sway_amplitude: float = np.deg2rad(12.0)
        self._antenna_frequency: float = 0.5
        self._breathing_start_time: float = 0.0
        self._breathing_scale: float = 1.0

        # Dance offset (added by dance modes)
        self._dance_lock = threading.Lock()
        self._dance_pose: Optional[NDArray] = None  # 4x4 matrix
        self._dance_antennas: Tuple[float, float] = (0.0, 0.0)
        self._dance_body_yaw: float = 0.0

        # Vocal offset (added by VocalMotionDriver)
        self._vocal_lock = threading.Lock()
        self._vocal_orientation: Optional[NDArray] = None  # [roll, pitch, yaw] radians
        self._vocal_position: Optional[NDArray] = None  # [x, y, z] meters

        logger.info("MotionController initialized")

    # ========================================================================
    # Dance Mode Interface
    # ========================================================================

    def add_dance_offset(
        self,
        position: NDArray,
        orientation: NDArray,
        antennas: Tuple[float, float],
        body_yaw: float,
    ) -> None:
        """Add dance motion offset (called by dance modes).

        Args:
            position: [x, y, z] offset in meters
            orientation: [roll, pitch, yaw] offset in radians
            antennas: (left, right) antenna positions
            body_yaw: Body yaw offset in radians

        """
        from reachy_mini import utils  # pyright: ignore[reportMissingImports]

        # Convert to 4x4 pose matrix
        dance_pose = utils.create_head_pose(
            *position,
            *orientation,
            degrees=False,
        )

        with self._dance_lock:
            self._dance_pose = dance_pose
            self._dance_antennas = antennas
            self._dance_body_yaw = body_yaw

    def clear_dance_offset(self) -> None:
        """Clear dance offset (when dance mode stops)."""
        with self._dance_lock:
            self._dance_pose = None
            self._dance_antennas = (0.0, 0.0)
            self._dance_body_yaw = 0.0

    # ========================================================================
    # Vocal Motion Interface
    # ========================================================================

    def add_vocal_offset(
        self,
        position: NDArray,
        orientation: NDArray,
        is_nodding: bool = False,
    ) -> None:
        """Add vocal response motion offset (called by VocalMotionDriver).

        Args:
            position: [x, y, z] offset in meters
            orientation: [roll, pitch, yaw] offset in radians
            is_nodding: Whether a speech nod is currently active.

        """
        # Inhibit face tracking during nods to prevent "smothering"/fighting
        # We extend the lock by cooldown to allow mechanical settling
        if is_nodding:
            self._inhibition_expiry = time.time() + self._inhibition_cooldown

        # Legacy flag support (though update_face_tracking now checks expiry)
        self._face_tracking_inhibited = is_nodding or (
            time.time() < self._inhibition_expiry
        )

        with self._vocal_lock:
            self._vocal_orientation = orientation
            self._vocal_position = position

    def clear_vocal_offset(self) -> None:
        """Clear vocal offset (when vocal_response = 0)."""
        with self._vocal_lock:
            self._vocal_orientation = None
            self._vocal_position = None

    # ========================================================================
    # Breathing Motion
    # ========================================================================

    def set_breathing_scale(self, scale: float) -> None:
        """Set breathing motion scale (0.0-1.0).

        Use 0.001 when listening to user speech.
        Use 1.0 for normal breathing.
        """
        self._breathing_scale = max(0.0, min(1.0, scale))

    def get_breathing_scale(self) -> float:
        """Get current breathing scale."""
        return self._breathing_scale

    def get_breathing_pose(self, t: float) -> Tuple[NDArray, Tuple[float, float]]:
        """Generate breathing pose at time t.

        Uses Lissajous figure: roll + y-sway with 90° phase offset.
        This creates the organic "figure-8" breathing pattern.

        Returns: (head_pose_4x4, (right_antenna, left_antenna))
        """
        breathing_time = t - self._breathing_start_time

        # Roll oscillation (scaled by breathing_scale)
        roll = (
            self._breathing_roll_amplitude
            * self._breathing_scale
            * np.sin(2 * np.pi * self._breathing_frequency * breathing_time)
        )

        # Y-sway with 90° phase offset for Lissajous figure
        y_sway = (
            self._breathing_y_amplitude
            * self._breathing_scale
            * np.sin(2 * np.pi * self._breathing_frequency * breathing_time + np.pi / 2)
        )

        # Create head pose with roll + y translation (pitch/yaw from face tracking)
        R_breathing = R.from_euler("xyz", [roll, 0, 0])
        head_pose = np.eye(4, dtype=np.float32)
        head_pose[:3, :3] = R_breathing.as_matrix()
        head_pose[1, 3] = y_sway  # Y translation for Lissajous gentle motion
        head_pose[2, 3] = 0.0  # Removed the redundant Z-lift to stay within kinematics

        # Antenna sway (opposite directions, also scaled)
        sway = (
            self._antenna_sway_amplitude
            * self._breathing_scale
            * np.sin(2 * np.pi * self._antenna_frequency * breathing_time)
        )
        antennas = (sway, -sway)

        return head_pose, antennas

    # ========================================================================
    # Face Tracking
    # ========================================================================

    def update_face_tracking(self, frame: NDArray) -> None:
        """Process frame for face tracking (called from camera loop)."""
        from reachy_mini.utils.interpolation import linear_pose_interpolation  # pyright: ignore[reportMissingImports]

        # Check inhibition (Nod Freeze)
        if time.time() < self._inhibition_expiry:
            return

        if self._face_tracking_inhibited:
            return

        if not self.enabled:
            return

        # Face tracking removed — this method is a no-op
        return

        if eye_center is not None:
            self._last_face_time = time.time()

            # Convert [-1,1] to pixels
            h, w = frame.shape[:2]
            px = int((eye_center[0] + 1) / 2 * w)
            py = int((eye_center[1] + 1) / 2 * h)

            try:
                # Get target pose from SDK IK (no actual movement)
                target_pose = self.mini.look_at_image(px, py, perform_movement=False)

                # Compute face tracking as OFFSET from neutral
                # Extract rotations
                R_target = R.from_matrix(target_pose[:3, :3])
                R_neutral = R.from_matrix(self._neutral_pose[:3, :3])

                # Relative rotation: how much to rotate from neutral to target
                R_offset = R_target * R_neutral.inv()

                # Translation offset
                # Note: target_pose from look_at_image has zero translation.
                # If we use it as-is, we pull the head to the origin (sinking it).
                # We only apply translation if the target actually provides it.
                t_offset = target_pose[:3, 3] - self._neutral_pose[:3, 3]
                if np.allclose(target_pose[:3, 3], 0):
                    t_offset = np.zeros(3, dtype=np.float32)

                # Build 4x4 offset matrix
                face_offset = np.eye(4, dtype=np.float32)
                face_offset[:3, :3] = R_offset.as_matrix().astype(np.float32)
                face_offset[:3, 3] = t_offset

                # Store face offset (thread-safe)
                with self._face_lock:
                    self._current_face_offset = face_offset.copy()
                    self._last_known_face_offset = face_offset.copy()
                    self._face_lost = False

                # DYNAMIC ANCHORING: Drift neutral pose towards the target
                # This establishes the user's face as the new "Zero" frame.
                # We interpolate Translation and Rotation separately.

                # We only anchor if the offset is not HUGE (sanity check)
                # and we are not inhibited (don't anchor to a nod!)
                if not self._face_tracking_inhibited:
                    # Drift rotation toward target, but keep translation at Home height
                    # to prevent the "Sinking Anchor" bug.
                    target_for_drift = target_pose.copy()
                    if np.allclose(target_pose[:3, 3], 0):
                        target_for_drift[:3, 3] = self._home_pose[:3, 3]

                    self._neutral_pose = linear_pose_interpolation(
                        self._neutral_pose.astype(np.float64),
                        target_for_drift.astype(np.float64),
                        self._anchor_drift_rate
                    )

            except Exception as e:
                logger.debug(f"look_at_image failed: {e}")

    def _update_pitch_interpolation(self, current_time: float) -> None:
        """Update smooth pitch interpolation."""
        if self._pitch_target is None:
            return

        elapsed = current_time - self._pitch_interp_start_time
        t = min(1.0, elapsed / self._pitch_interp_duration)

        self._current_pitch = lerp(self._pitch_interp_start, self._pitch_target, t)

        if t >= 1.0:
            self._pitch_target = None

    def _get_neutral_face_pose(self) -> NDArray:
        """Return neutral head pose when no face detected."""
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = -0.005
        pose[2, 3] = 0.015
        return pose

    # ========================================================================
    # Euler Composition
    # ========================================================================

    def compose_poses(
        self,
        breathing_pose: NDArray,
        face_offset: Optional[NDArray],
        dance_pose: Optional[NDArray] = None,
        vocal_orientation: Optional[NDArray] = None,
    ) -> NDArray:
        """Compose poses treating Face as Base and others as Local Offsets.

        Order:
        1. Base = Neutral Pose
        2. + Face Tracking (World Frame Offset) -> establishes "Look At" orientation
        3. + Vocal (Euler Pitch Offset) -> applied to Base Orientation
        4. + Breathing (Local Offset) -> rolls relative to "Look At"
        5. + Dance (Local Offset) -> rolls relative to "Look At"
        """
        from reachy_mini.utils.interpolation import compose_world_offset  # pyright: ignore[reportMissingImports]

        # 1. Establish Base Pose (Neutral + Face Tracking)
        # Start with standard neutral pose (includes Z-lift)
        base_pose = self._neutral_pose.copy()

        # Apply face offset (World Frame composition is used here as Face Tracking
        # is calculated as a rotation from Neutral in World Frame)
        if face_offset is not None:
            base_pose = compose_world_offset(
                base_pose.astype(np.float64),
                face_offset.astype(np.float64),
                reorthonormalize=True
            )

        # 1.5. FORCE HORIZON (Fix for "Weird Roll" when looking Up/Side)
        # The IK solver (look_at_image) sometimes introduces unwanted roll.
        # We explicitly strip it here to ensure the head stays level.
        R_base_raw = R.from_matrix(base_pose[:3, :3])
        yaw, pitch, roll = R_base_raw.as_euler("ZYX", degrees=False)

        # Force Roll to 0 (Keep Head Level)
        # We allow Pitch (Looking Up/Down) and Yaw (Looking Left/Right)
        base_pose[:3, :3] = R.from_euler(
            "ZYX", [yaw, pitch, 0.0], degrees=False
        ).as_matrix()

        # 2. Apply Vocal Pitch (Nod) via Euler Addition
        # CRITICAL FIX: We apply pitch analytically to the ZYX Euler stack.
        # Matrix composition caused nods to become rolls when head was yawed 90deg.
        # Intrinsic ZYX decomposition ensures we always rotate around the Head's Lateral (Y) axis.
        if vocal_orientation is not None:
            # Current Base Orientation
            R_base = R.from_matrix(base_pose[:3, :3])
            yaw, pitch, roll = R_base.as_euler("ZYX", degrees=False)

            # Add Vocal Pitch (index 1)
            # Assuming +Pitch is Down/Forward in Intrinsic Frame
            pitch += vocal_orientation[1]

            # Reconstruct Rotation
            R_new = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False)
            base_pose[:3, :3] = R_new.as_matrix()

            # Note: We ignore vocal_position (translation) as requested to prevent IK collisions

            # 2.5. Compensate for Nod Arc (X-Retraction)
            # When pitching down (+pitch), the head naturally arcs forward.
            # We subtract X to keep the face plane cleaner/simpler.
            # Factor 0.06 = 6cm retraction per radian (~1mm per degree)
            # Only apply if pitch is positive (looking down)
            if vocal_orientation[1] > 0:
                x_retract = vocal_orientation[1] * 0.06
                base_pose[0, 3] -= x_retract

        # 3. Accumulate Local Offsets (Breathing + Dance)
        local_offset = np.eye(4, dtype=np.float32)

        # Breathing (Roll + Sway) applies to the nodded head
        local_offset = local_offset @ breathing_pose

        if dance_pose is not None:
            local_offset = local_offset @ dance_pose

        # 4. Apply Local Offsets to Base Pose
        combined_pose = base_pose @ local_offset

        return combined_pose

    # ========================================================================
    # Body Anchor System
    # ========================================================================

    def apply_body_follow(self, head_yaw: float, current_time: float) -> float:
        """Apply body anchor state machine.

        Args:
            head_yaw: Current head yaw in radians
            current_time: Current time

        Returns:
            Body yaw in radians

        """
        head_yaw_deg = np.rad2deg(head_yaw)

        if self._anchor_state == AnchorState.ANCHORED:
            # Check strain between head and body
            strain = head_yaw_deg - self._body_anchor_yaw
            # Normalize to [-180, 180]
            while strain > 180:
                strain -= 360
            while strain < -180:
                strain += 360

            if abs(strain) > self._strain_threshold:
                # Start syncing
                self._anchor_state = AnchorState.SYNCING
                self._sync_start_time = current_time
                self._sync_start_yaw = self._body_anchor_yaw
                self._sync_target_yaw = head_yaw_deg
                logger.debug(
                    f"Body anchor: ANCHORED -> SYNCING (strain={strain:.1f}deg)"
                )

            return np.deg2rad(self._body_anchor_yaw)

        elif self._anchor_state == AnchorState.SYNCING:
            # Interpolate body toward head
            elapsed = current_time - self._sync_start_time
            t = min(1.0, elapsed / self._sync_duration)
            t_eased = ease_in_out(t)

            body_yaw_deg = lerp(self._sync_start_yaw, self._sync_target_yaw, t_eased)

            if t >= 1.0:
                # Sync complete - start stabilizing
                self._anchor_state = AnchorState.STABILIZING
                self._stability_start_time = current_time
                self._last_head_yaw = head_yaw_deg
                logger.debug("Body anchor: SYNCING -> STABILIZING")

            return np.deg2rad(body_yaw_deg)

        elif self._anchor_state == AnchorState.STABILIZING:
            # Body follows head, waiting for head to stabilize
            if abs(head_yaw_deg - self._last_head_yaw) > self._stability_threshold:
                # Head still moving - reset timer
                self._stability_start_time = current_time
                self._last_head_yaw = head_yaw_deg

            elapsed = current_time - self._stability_start_time
            if elapsed >= self._stability_duration:
                # Head stable - create new anchor
                self._body_anchor_yaw = head_yaw_deg
                self._anchor_state = AnchorState.ANCHORED
                logger.debug(
                    f"Body anchor: STABILIZING -> ANCHORED (anchor={head_yaw_deg:.1f}deg)"
                )

            return np.deg2rad(head_yaw_deg)

        return 0.0

    # ========================================================================
    # Control Loop
    # ========================================================================

    def start(self) -> None:
        """Start the motion control loop."""
        if self._running:
            return

        self._running = True
        self._breathing_start_time = time.time()
        self._thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._thread.start()
        logger.info("Motion control loop started")

    def stop(self) -> None:
        """Stop the motion control loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Motion control loop stopped")

    def _motion_loop(self) -> None:
        """100Hz motion control loop."""
        loop_interval = 0.01  # 100Hz

        while self._running:
            start_time = time.time()

            if self.enabled:
                try:
                    self._motion_tick(start_time)
                except Exception as e:
                    logger.error(f"Motion tick error: {e}")

            # Maintain loop timing
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _motion_tick(self, current_time: float) -> None:
        """Single motion update tick."""
        # Get breathing pose
        breathing_pose, breathing_antennas = self.get_breathing_pose(current_time)

        # Get face tracking offset (thread-safe)
        with self._face_lock:
            face_offset = (
                self._current_face_offset.copy()
                if self._current_face_offset is not None
                else None
            )
            last_known = (
                self._last_known_face_offset.copy()
                if self._last_known_face_offset is not None
                else None
            )

        # Check for face lost timeout - retain last known offset
        if (
            face_offset is not None
            and (current_time - self._last_face_time) > self._face_lost_timeout
        ):
            if not self._face_lost:
                logger.debug("Face lost - retaining last known offset")
                self._face_lost = True

            # If face is lost, slowly drift Neutral back to Home
            from reachy_mini.utils.interpolation import linear_pose_interpolation #type: ignore[reportMissingImports]


            self._neutral_pose = linear_pose_interpolation(
                self._neutral_pose.astype(np.float64),
                self._home_pose.astype(np.float64),
                self._return_to_home_rate
            )

            # Recalculate offset based on drifting neutral to keep head steady-ish
            # If Neutral moves, and Offset is fixed, the head moves.
            # This effectively "turns head back to center" slowly.)
            face_offset = last_known

        # Get dance offset (thread-safe)
        with self._dance_lock:
            dance_pose = (
                self._dance_pose.copy() if self._dance_pose is not None else None
            )
            dance_antennas = self._dance_antennas
            dance_body_yaw = self._dance_body_yaw

        # Get vocal offset (thread-safe)
        with self._vocal_lock:
            vocal_orientation = (
                self._vocal_orientation.copy()
                if self._vocal_orientation is not None
                else None
            )

        # Compose breathing + face tracking + dance + vocal
        combined_pose = self.compose_poses(
            breathing_pose, face_offset, dance_pose, vocal_orientation
        )

        # Extract head yaw for body anchor
        R_combined = R.from_matrix(combined_pose[:3, :3])
        _, _, head_yaw = R_combined.as_euler("xyz", degrees=False)

        # Apply body anchor system (add dance body yaw offset)
        body_yaw = self.apply_body_follow(head_yaw, current_time) + dance_body_yaw

        # Combine antennas: breathing + dance
        final_antennas = [
            breathing_antennas[0] + dance_antennas[0],
            breathing_antennas[1] + dance_antennas[1],
        ]

        # Send to robot
        try:
            self.mini.set_target(
                head=combined_pose,
                antennas=final_antennas,
                body_yaw=body_yaw,
            )
        except Exception as e:
            if (
                not hasattr(self, "_last_set_target_error")
                or (current_time - self._last_set_target_error) > 1.0
            ):
                logger.error(f"set_target failed: {e}")
                self._last_set_target_error = current_time

    def reset(self) -> None:
        """Reset to neutral position."""
        self._anchor_state = AnchorState.ANCHORED
        self._body_anchor_yaw = 0.0
        self.clear_dance_offset()
        self.clear_vocal_offset()
        self._neutral_pose = self._home_pose.copy()  # Reset Anchor
        with self._face_lock:
            self._current_face_offset = None
            self._last_known_face_offset = None
