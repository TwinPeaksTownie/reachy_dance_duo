"""Default configuration for Reachy Dance Suite.

These values can be overridden via CLI args, config file, or live UI tuning.
"""

import numpy as np

from .core.safety_mixer import SafetyConfig


def get_default_safety_config() -> SafetyConfig:
    """Get default SafetyConfig with safe, tested values."""
    return SafetyConfig(
        # Inverse collision limiter
        z_threshold=0.005,  # 5mm - below this, pitch limiting kicks in
        max_pitch_at_low_z=0.15,  # ~8.6 deg - max downward pitch when Z is low
        # Smoothing
        smoothing_alpha=1.0,  # No smoothing by default - user can lower via UI
        # Position limits (meters)
        max_position=np.array([0.05, 0.05, 0.04]),
        min_position=np.array([-0.05, -0.05, -0.03]),
        # Orientation limits (radians)
        max_orientation=np.array([0.5, 0.4, 0.7]),  # ~28, 23, 40 deg
        min_orientation=np.array([-0.5, -0.4, -0.7]),
        # Antenna limits (radians)
        max_antenna=3.15,
        min_antenna=-3.15,
        # Body yaw limits (radians)
        max_body_yaw=0.8,  # ~46 deg
        min_body_yaw=-0.8,
        # Global intensity
        intensity=1.0,
    )


# App configuration
APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 9000,
    "debug": False,
}

# Audio configuration (16kHz - Reachy Mini Audio hardware limit)
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 512,  # ~32ms latency at 16kHz
}

# Mode-specific defaults
MODE_A_CONFIG = {
    "bpm_min": 70.0,
    "bpm_max": 140.0,
    "noise_calibration_duration": 14.0,
    "dance_noise_calibration_duration": 8.0,
    "beats_per_sequence": 8,
}

MODE_B_CONFIG = {
    "download_dir": "downloads",
    "output_dir": "output",
}

MODE_C_CONFIG = {
    # FFT frequency bands
    "bass_range": (20, 150),  # Hz
    "vocal_range": (300, 3000),  # Hz
    "high_range": (6000, None),  # Hz (None = to max)
    # Movement mapping
    "max_yaw": 0.75,  # Body sway ~43 deg
    "max_pitch": 0.30,  # Head nod ~17 deg
    "max_z": 0.015,  # Vertical bounce 1.5cm
    # Physics smoothing (asymmetric attack/decay)
    "physics": {
        "body": {"attack": 0.2, "decay": 0.12},
        "head": {"attack": 0.3, "decay": 0.1},
        "ant": {"attack": 0.6, "decay": 0.35},
        "z": {"attack": 0.8, "decay": 0.15},
    },
}

# YouTube Music configuration
YTMUSIC_CONFIG = {
    "client_id": "",  # Set via env var YTMUSIC_CLIENT_ID
    "client_secret": "",  # Set via env var YTMUSIC_CLIENT_SECRET
}
