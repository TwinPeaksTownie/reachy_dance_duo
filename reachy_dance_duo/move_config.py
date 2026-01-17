"""Shared move dampening and mirroring configuration.

This module holds per-move amplitude overrides and Y-axis mirror settings
that can be tuned live via the web UI.
Defaults are loaded from JSON files for easy editing.
"""

import json
from pathlib import Path
from typing import cast

# Load defaults from JSON
_CONFIG_PATH = Path(__file__).parent / "move_dampening.json"
_MIRROR_PATH = Path(__file__).parent / "move_mirror.json"

# Moves that support Y-axis mirroring (lateral movements)
# Expanded from 7 to 13 moves
MIRRORABLE_MOVES = [
    "dizzy_spin",
    "grid_snap",
    "head_tilt_roll",
    "interwoven_spirals",
    "jackson_square",
    "pendulum_swing",
    "polyrhythm_combo",
    "sharp_side_tilt",
    "side_glance_flick",
    "side_peekaboo",
    "side_to_side_sway",
    "stumble_and_recover",
    "uh_huh_tilt",
]


def _load_defaults() -> dict[str, float]:
    """Load default dampening values from JSON file."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return cast(dict[str, float], json.load(f))
    return {}


def _load_mirror_defaults() -> dict[str, bool]:
    """Load mirror settings from JSON file."""
    if _MIRROR_PATH.exists():
        with open(_MIRROR_PATH) as f:
            return cast(dict[str, bool], json.load(f))
    # Default: no moves mirrored
    return {move: False for move in MIRRORABLE_MOVES}


def _save_to_file() -> None:
    """Save current dampening values to JSON file."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(_move_dampening, f, indent=2, sort_keys=True)
        f.write("\n")


def _save_mirror_to_file() -> None:
    """Save current mirror settings to JSON file."""
    with open(_MIRROR_PATH, "w") as f:
        json.dump(_move_mirror, f, indent=2, sort_keys=True)
        f.write("\n")


DEFAULT_MOVE_DAMPENING = _load_defaults()

# Runtime state - starts with defaults, can be modified via API
_move_dampening: dict[str, float] = DEFAULT_MOVE_DAMPENING.copy()
_move_mirror: dict[str, bool] = _load_mirror_defaults()
_all_moves: list[str] = []


def init_moves(move_names: list[str]) -> None:
    """Initialize with full list of available moves."""
    global _all_moves
    _all_moves = sorted(move_names)


def get_all_moves() -> list[str]:
    """Get list of all available move names."""
    return _all_moves.copy()


def get_dampening(move_name: str) -> float:
    """Get dampening value for a move (defaults to 1.0 if not set)."""
    return _move_dampening.get(move_name, 1.0)


def set_dampening(move_name: str, value: float) -> None:
    """Set dampening value for a move."""
    _move_dampening[move_name] = max(0.0, min(2.0, value))  # Clamp 0-2


def get_all_dampening() -> dict[str, float]:
    """Get dampening values for all moves."""
    result = {}
    for move in _all_moves:
        result[move] = _move_dampening.get(move, 1.0)
    return result


def set_all_dampening(values: dict[str, float]) -> None:
    """Set multiple dampening values at once and save to file."""
    for move, value in values.items():
        set_dampening(move, value)
    _save_to_file()


def reset_to_defaults() -> None:
    """Reset all dampening to default values."""
    global _move_dampening
    _move_dampening = _load_defaults()


# Mirror functions
def get_mirrorable_moves() -> list[str]:
    """Get list of moves that support Y-axis mirroring."""
    return MIRRORABLE_MOVES.copy()


def is_mirrored(move_name: str) -> bool:
    """Check if a move is set to mirror Y-axis."""
    return _move_mirror.get(move_name, False)


def set_mirror(move_name: str, mirrored: bool) -> None:
    """Set mirror state for a move."""
    if move_name in MIRRORABLE_MOVES:
        _move_mirror[move_name] = mirrored


def get_all_mirror() -> dict[str, bool]:
    """Get mirror state for all mirrorable moves."""
    return {move: _move_mirror.get(move, False) for move in MIRRORABLE_MOVES}


def set_all_mirror(values: dict[str, bool]) -> None:
    """Set multiple mirror values at once and save to file."""
    for move, mirrored in values.items():
        set_mirror(move, mirrored)
    _save_mirror_to_file()


def reset_mirror_to_defaults() -> None:
    """Reset all mirror settings to defaults (all False)."""
    global _move_mirror
    _move_mirror = {move: False for move in MIRRORABLE_MOVES}
    _save_mirror_to_file()
