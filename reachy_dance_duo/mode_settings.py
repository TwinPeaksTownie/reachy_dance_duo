"""Per-mode amplitude and intensity settings.

Stores settings for each dance mode that persist across restarts.
Settings can be tuned live via the web UI and are saved to JSON.
"""

import json
from pathlib import Path
from typing import cast

# Load defaults from JSON
_CONFIG_PATH = Path(__file__).parent / "mode_settings.json"

# Default values if JSON doesn't exist
_DEFAULTS = {
    "live_groove": {
        "intensity": 1.0,
        "volume_gate_threshold": 0.005,  # Volume threshold for detecting music
        "bpm_stability_threshold": 8.0,  # Max std dev for BPM to be considered locked
    },
    "beat_bandit": {
        "amplitude_scale": 0.5,
        "interpolation_alpha": 0.3,
        "antenna_energy_threshold": 0.25,
        "antenna_sensitivity": 0.6,
        "antenna_amplitude": 1.75,
        "antenna_gain": 8.0,
    },
    "synthwave_serenade": {
        "intensity": 1.0,
        "liveliness": 0.9,
        "strain_threshold": 15.0,
        "vocal_response": 1.0,  # 0.01-1.0, always enabled
        "playback_volume": 1.5,  # Audio playback gain multiplier
    },
}


def _load_from_file() -> dict[str, dict[str, float]]:
    """Load settings from JSON file."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return cast(dict[str, dict[str, float]], json.load(f))
    return _DEFAULTS.copy()


def _save_to_file() -> None:
    """Save current settings to JSON file."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(_settings, f, indent=2)
        f.write("\n")


# Runtime state - starts with file values
_settings: dict[str, dict[str, float]] = _load_from_file()


def get_mode_settings(mode_id: str) -> dict[str, float]:
    """Get all settings for a mode."""
    mode_key = mode_id.lower()
    if mode_key in _settings:
        return _settings[mode_key].copy()
    return {}


def get_setting(mode_id: str, key: str) -> float:
    """Get a specific setting for a mode."""
    mode_key = mode_id.lower()
    if mode_key in _settings and key in _settings[mode_key]:
        return _settings[mode_key][key]
    # Return default if available
    if mode_key in _DEFAULTS and key in _DEFAULTS[mode_key]:
        return _DEFAULTS[mode_key][key]
    return 1.0  # Fallback


def set_setting(mode_id: str, key: str, value: float) -> None:
    """Set a specific setting for a mode."""
    mode_key = mode_id.lower()
    if mode_key not in _settings:
        _settings[mode_key] = {}
    _settings[mode_key][key] = value


def update_mode_settings(mode_id: str, updates: dict[str, float]) -> None:
    """Update multiple settings for a mode and save to file."""
    mode_key = mode_id.lower()
    if mode_key not in _settings:
        _settings[mode_key] = {}
    _settings[mode_key].update(updates)
    _save_to_file()


def get_all_settings() -> dict[str, dict[str, float]]:
    """Get all settings for all modes."""
    return {k: v.copy() for k, v in _settings.items()}


def sync_from_file() -> dict[str, dict[str, float]]:
    """Reload settings from JSON file (for sync button)."""
    global _settings
    _settings = _load_from_file()
    return get_all_settings()


def reset_to_defaults() -> dict[str, dict[str, float]]:
    """Reset all settings to defaults and save."""
    global _settings
    _settings = {k: v.copy() for k, v in _DEFAULTS.items()}
    _save_to_file()
    return get_all_settings()
