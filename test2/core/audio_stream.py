"""AudioStream - Platform-agnostic audio input abstraction.

Handles the differences between:
- Live Groove: Reachy Mini Audio (USB microphone on the robot)
- Bluetooth Streamer: BlackHole loopback (Mac system audio capture)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np  # type: ignore

try:
    import pyaudio  # type: ignore
except ImportError:
    pyaudio = None

PYAUDIO_AVAILABLE = pyaudio is not None


class AudioStream:
    """Platform-agnostic audio input stream.

    Wraps PyAudio to provide a unified interface for different
    audio sources (microphone, loopback, etc.).
    """

    # Known device patterns
    DEVICE_PATTERNS = {
        "live_groove": "Reachy Mini Audio",  # Robot's USB microphone
        "bluetooth_streamer": "BlackHole",  # Legacy name
        "synthwave_serenade": "BlackHole",  # New name for loopback
    }

    def __init__(
        self,
        device_pattern: str,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
    ):
        """Initialize audio stream.

        Args:
            device_pattern: Name pattern to search for (e.g., "BlackHole")
            sample_rate: Audio sample rate in Hz
            chunk_size: Samples per buffer
            channels: Number of audio channels (1=mono, 2=stereo)

        """
        self.device_pattern = device_pattern
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels

        self.pa: Optional["pyaudio.PyAudio"] = None
        self.stream: Optional["pyaudio.Stream"] = None
        self.callback: Optional[Callable[[np.ndarray], None]] = None
        self.running = False
        self.device_index: Optional[int] = None

    @classmethod
    def create_for_mode(
        cls, mode: str, sample_rate: int = 16000, chunk_size: int = 512
    ) -> AudioStream:
        """Create stream for a specific mode.

        Args:
            mode: "live_groove" for microphone, "bluetooth_streamer" for loopback
            sample_rate: Audio sample rate in Hz
            chunk_size: Samples per buffer

        Returns:
            Configured AudioStream instance

        """
        mode_key = mode.lower()
        pattern = cls.DEVICE_PATTERNS.get(mode_key)
        if not pattern:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'live_groove', 'synthwave_serenade', or 'bluetooth_streamer'."
            )

        # Live Groove (mic) uses stereo input from Reachy Mini Audio
        channels = 2 if mode_key == "live_groove" else 1

        return cls(
            device_pattern=pattern,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channels=channels,
        )

    def find_device(self) -> Optional[int]:
        """Find audio device by name pattern.

        Returns:
            Device index if found, None otherwise

        """
        if pyaudio is None:
            print("[AudioStream] PyAudio not installed.")
            return None

        if self.pa is None:
            self.pa = pyaudio.PyAudio()

        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = info.get("name", "")
            max_inputs = info.get("maxInputChannels", 0)

            if max_inputs > 0 and self.device_pattern in name:
                print(f"[AudioStream] Found device: [{i}] {name}")
                return i

        print(f"[AudioStream] Device not found: {self.device_pattern}")
        return None

    def start(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start the audio stream.

        Args:
            callback: Function called with each audio buffer (numpy array)

        Returns:
            True if started successfully, False otherwise

        """
        if self.running:
            return True

        self.callback = callback

        if pyaudio is None:
            print("[AudioStream] Cannot start: PyAudio not installed")
            return False

        if self.pa is None:
            self.pa = pyaudio.PyAudio()

        self.device_index = self.find_device()
        if self.device_index is None:
            print(
                f"[AudioStream] Cannot start: device '{self.device_pattern}' not found"
            )
            return False

        try:
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )
            self.stream.start_stream()
            self.running = True
            print(f"[AudioStream] Started listening on {self.device_pattern}")
            return True

        except Exception as e:
            print(f"[AudioStream] Failed to start: {e}")
            return False

    def stop(self, terminate_pa: bool = True) -> None:
        """Stop the audio stream.

        Args:
            terminate_pa: If True, also terminate the PyAudio instance.

        """
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if terminate_pa:
            self.terminate()

        print("[AudioStream] Stopped")

    def terminate(self) -> None:
        """Terminate the PyAudio instance."""
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = None
            print("[AudioStream] Terminated PyAudio")

    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int,
    ) -> tuple[None, int]:
        """PyAudio callback - processes incoming audio data."""
        if not self.running or self.callback is None:
            return (None, pyaudio.paContinue)

        # Convert bytes to numpy array
        samples = np.frombuffer(in_data, dtype=np.float32)

        # Convert stereo to mono if needed (for Reachy Mini Audio)
        if self.channels == 2:
            samples = ((samples[0::2] + samples[1::2]) / 2.0).astype(np.float32)

        # Call user callback
        try:
            self.callback(samples)
        except Exception as e:
            print(f"[AudioStream] Callback error: {e}")

        return (None, pyaudio.paContinue)

    @staticmethod
    def list_devices() -> list[dict]:
        """List all available audio input devices.

        Returns:
            List of device info dictionaries

        """
        if pyaudio is None:
            return []

        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info.get("name"),
                        "channels": info.get("maxInputChannels"),
                        "sample_rate": int(info.get("defaultSampleRate", 0)),
                    }
                )

        pa.terminate()
        return devices
