"""Dance mode behaviors."""

from .base import DanceMode
from .live_groove import LiveGroove
from .synthwave_serenade import (
    SynthwaveSerenade,
    DiscoAudioDancer,
    SystemAudioDancer,
    BluetoothStreamer,
)  # Legacy aliases
from .connected_choreographer import ConnectedChoreographer

__all__ = [
    "DanceMode",
    "LiveGroove",
    "SynthwaveSerenade",
    "DiscoAudioDancer",
    "SystemAudioDancer",
    "BluetoothStreamer",
    "ConnectedChoreographer",
]
