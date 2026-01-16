"""
Base DanceMode - Abstract base class for all dance behaviors.

All modes (Live Groove, Bluetooth Streamer, Connected Choreographer)
inherit from this class and implement the start/stop/get_status interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


class DanceMode(ABC):
    """Abstract base class for dance modes.

    Modes generate movement intent and send it to the SafetyMixer.
    They should never call mini.set_target() directly.
    """

    # Mode identifier (set by subclasses)
    MODE_ID: str = "base"
    MODE_NAME: str = "Base Mode"

    def __init__(self, safety_mixer: SafetyMixer):
        """Initialize the dance mode.

        Args:
            safety_mixer: The SafetyMixer instance to send intents to
        """
        self.mixer = safety_mixer
        self.running = False
        self._status: dict[str, Any] = {}

    @abstractmethod
    async def start(self) -> None:
        """Start the dance mode.

        This should begin audio processing and movement generation.
        Implementations must set self.running = True.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the dance mode.

        This should stop audio processing and return to idle.
        Implementations must set self.running = False.
        """
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get current mode status for UI display.

        Returns:
            Dictionary with mode-specific status info.
            Should include at minimum:
            - "mode": MODE_ID
            - "running": bool
            - "state": str (e.g., "idle", "dancing", "analyzing")
        """
        pass

    def is_running(self) -> bool:
        """Check if the mode is currently running."""
        return self.running
