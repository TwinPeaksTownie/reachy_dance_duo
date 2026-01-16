"""
Mode B: Choreographer - Pre-analyzed YouTube video dance mode.

Downloads audio from YouTube, performs offline Librosa analysis,
and generates choreographed dance sequences based on song energy.

This is the "Performer" - knows the song in advance, can anticipate
changes, and delivers rehearsed movements.

Key features:
- YouTube download via yt-dlp
- Full-song Librosa analysis (tempo, beats, onsets, energy)
- Energy-based 8-beat sequence selection (high/medium/low)
- Audio playback synchronization
- Pre-computed choreography for smooth transitions
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import threading
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import librosa
import numpy as np

from .base import DanceMode
from ..core.safety_mixer import MovementIntent
from .. import mode_settings

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


# 8-beat dance sequences organized by energy level
# Format: [x, y, z, roll, pitch, yaw] in (cm, cm, cm, deg, deg, deg)
EIGHT_BEAT_SEQUENCES = {
    "high_energy": [
        # Sharp left/right snaps
        [
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
            {"name": "Sharp snap left", "coords": [0, 0.5, 0.25, -15, 0, -24]},
            {"name": "Sharp snap right", "coords": [0, 0.5, 0.25, 15, 0, 24]},
        ],
        # Up/down power nods
        [
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
            {"name": "Strong head drop", "coords": [0, 0, -1.9, 0, -22, 0]},
            {"name": "Head snap up", "coords": [0, 0, 2.2, 0, 19, 0]},
        ],
        # Forward/back thrust
        [
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
            {"name": "Head thrust forward", "coords": [0, 2.6, 0, 0, -9, 0]},
            {"name": "Head jerk back", "coords": [0, -1.9, 0, 0, 11, 0]},
        ],
        # Aggressive circular
        [
            {"name": "Head whip left", "coords": [0, 0, 0, -24, 0, -27]},
            {"name": "Head slam down", "coords": [0, 1.1, -1.9, 0, -26, 0]},
            {"name": "Head whip right", "coords": [0, 0, 0, 24, 0, 27]},
            {"name": "Head throw back", "coords": [0, -1.5, 1.9, 0, 21, 0]},
            {"name": "Diagonal tilt left", "coords": [0, 0.8, 0, -24, -11, -15]},
            {"name": "Power nod center", "coords": [0, 0, -1.1, 0, -24, 0]},
            {"name": "Diagonal tilt right", "coords": [0, -0.8, 0, 24, -11, 15]},
            {"name": "Head explosion up", "coords": [0, 0, 2.6, 0, 24, 0]},
        ],
    ],
    "medium_energy": [
        # Flowing sway
        [
            {"name": "Flow left", "coords": [0, 1.1, 0.4, -15, -9, -19]},
            {"name": "Flow center up", "coords": [0, 0, 1.1, 0, 11, 0]},
            {"name": "Flow right", "coords": [0, 0.8, 0.4, 15, -6, 19]},
            {"name": "Flow back center", "coords": [0, -0.8, 0.8, 0, 15, 0]},
            {"name": "Flow diagonal 1", "coords": [0, 1.5, 0, -19, -11, -15]},
            {"name": "Flow diagonal 2", "coords": [0, -0.8, 1.5, 11, 15, 11]},
            {"name": "Flow circle left", "coords": [0, 0, 0.8, -11, -4, -22]},
            {"name": "Flow circle right", "coords": [0, 0, 0.8, 11, -4, 22]},
        ],
        # Smooth waves
        [
            {"name": "Wave left start", "coords": [0, 0.8, 0, -13, -6, -15]},
            {"name": "Wave center dip", "coords": [0, 0, -0.8, 0, -11, 0]},
            {"name": "Wave right rise", "coords": [0, 0.8, 0.8, 13, 8, 15]},
            {"name": "Wave back center", "coords": [0, -0.8, 0, 0, 9, 0]},
            {"name": "Wave forward left", "coords": [0, 1.5, 0, -11, -8, -11]},
            {"name": "Wave up right", "coords": [0, 0, 1.5, 11, 11, 11]},
            {"name": "Wave down left", "coords": [0, -0.8, -0.8, -9, -9, -13]},
            {"name": "Wave reset center", "coords": [0, 0.8, 0.8, 0, 6, 0]},
        ],
        # Alternating emphasis
        [
            {"name": "Emphasis left nod", "coords": [0, 0, 0, -15, -13, -19]},
            {"name": "Soft center up", "coords": [0, 0, 0.8, 0, 8, 0]},
            {"name": "Emphasis right nod", "coords": [0, 0, 0, 15, -13, 19]},
            {"name": "Soft center up", "coords": [0, 0, 0.8, 0, 8, 0]},
            {"name": "Strong forward", "coords": [0, 2.2, 0, 0, -16, 0]},
            {"name": "Gentle back", "coords": [0, -0.8, 0.8, 0, 6, 0]},
            {"name": "Side emphasis left", "coords": [0, 0.8, 0, -19, 0, -15]},
            {"name": "Side emphasis right", "coords": [0, 0.8, 0, 19, 0, 15]},
        ],
        # Figure-8 pattern
        [
            {"name": "Figure-8 start", "coords": [0, 1, 1, -15, -10, -18]},
            {"name": "Figure-8 cross center", "coords": [0, 0, 0, 0, 0, 0]},
            {"name": "Figure-8 right loop", "coords": [0, 1, 1, 15, 10, 18]},
            {"name": "Figure-8 back cross", "coords": [0, -1, -1, 0, 5, 0]},
            {"name": "Figure-8 left down", "coords": [0, 0, -1, -18, -15, -20]},
            {"name": "Figure-8 up cross", "coords": [0, 1, 1, 0, 12, 0]},
            {"name": "Figure-8 right down", "coords": [0, 0, -1, 18, -15, 20]},
            {"name": "Figure-8 complete", "coords": [0, -1, 0, 0, 8, 0]},
        ],
    ],
    "low_energy": [
        # Gentle nod
        [
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -22, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 18, 0]},
        ],
        # Soft turn
        [
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
        ],
        # Gentle sway
        [
            {"name": "Gentle nod down", "coords": [0, 0, 0, 0, -12, 0]},
            {"name": "Gentle nod up", "coords": [0, 0, 0, 0, 8, 0]},
            {"name": "Soft turn left", "coords": [0, 0, 0, 0, 0, -18]},
            {"name": "Soft turn right", "coords": [0, 0, 0, 0, 0, 18]},
            {"name": "Light tilt left", "coords": [0, 0, 0, -15, 0, 0]},
            {"name": "Light tilt right", "coords": [0, 0, 0, 15, 0, 0]},
            {"name": "Gentle pitch forward", "coords": [0, 0, 0, 0, -8, 0]},
            {"name": "Gentle pitch back", "coords": [0, 0, 0, 0, 6, 0]},
        ],
        # Contemplative moves
        [
            {"name": "Thoughtful nod", "coords": [0, 0, 0, 0, -10, 0]},
            {"name": "Curious tilt left", "coords": [0, 0, 0, -8, 0, -12]},
            {"name": "Ponder left turn", "coords": [0, 0, 0, -6, 0, -16]},
            {"name": "Ponder right turn", "coords": [0, 0, 0, 6, 0, 16]},
            {"name": "Meditative tilt right", "coords": [0, 0, 0, 12, -2, 10]},
            {"name": "Peaceful center", "coords": [0, 0, 0, 0, 0, 0]},
            {"name": "Gentle roll left", "coords": [0, 0, 0, -10, 2, -8]},
            {"name": "Gentle roll right", "coords": [0, 0, 0, 10, 2, 8]},
        ],
    ],
}


@dataclass
class SongAnalysis:
    """Results of Librosa song analysis."""

    audio_path: str
    duration: float
    tempo: float
    beat_times: np.ndarray
    energy_per_beat: np.ndarray  # 0-1 energy at each beat
    sequence_assignments: list[str]  # "high"/"medium"/"low" per 8-beat block
    # Continuous signal envelopes
    energy_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    onset_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    envelope_sr: int = 22050  # Sample rate of the envelope (usually audio_sr / hop_length)
    hop_length: int = 512


@dataclass
class ChoreographerConfig:
    """Configuration for Choreographer mode."""

    download_dir: str = "downloads"
    output_dir: str = "output"
    amplitude_scale: float = 0.5  # Global movement scale
    interpolation_alpha: float = 0.3  # Smoothing between beat poses

    # Energy thresholds
    # Antenna Control
    antenna_sensitivity: float = 1.0  # Multiplier for antenna responsiveness (Slider A)
    antenna_amplitude: float = 3.15   # Max travel for antenna movement (Default to Full Range)
    antenna_gain: float = 20.0        # Fixed pre-amplification for raw RMS signal

    # Breathing motion (between choreographed movements)
    breathing_y_amplitude: float = 0.016  # Y sway amplitude (meters)
    breathing_y_freq: float = 0.2  # Y sway frequency (Hz)
    breathing_roll_amplitude: float = 0.222  # Roll amplitude (radians)
    breathing_roll_freq: float = 0.15  # Roll frequency (Hz)

    # Antenna beat drops - antennas stay up near 0, drop hard on energetic beats
    antenna_rest_position: float = -0.1  # Resting position (near top)
    antenna_drop_max: float = 2.0  # Maximum drop magnitude (left goes to -2.0, right to +2.0)
    antenna_energy_threshold: float = 0.25  # Energy threshold to trigger drop (0-1) - lower = more drops
    antenna_decay_rate: float = 4.0  # How fast antennas spring back (lower = longer visible drops)

    # Neutral pose for breathing reference
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.01]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))


class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def download_audio(self, url: str) -> Optional[str]:
        """Download audio from YouTube URL.

        Returns path to downloaded WAV file or None if failed.
        """
        try:
            import yt_dlp
        except ImportError:
            print("[Choreographer] yt-dlp not installed")
            return None

        output_template = str(self.download_dir / "%(title)s [%(id)s].%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "extract_flat": False,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "320",
                }
            ],
            "postprocessor_args": ["-loglevel", "error"],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Handle search results (playlist)
                if "entries" in info:
                    if not info["entries"]:
                        print(f"[Choreographer] No search results for: {url}")
                        return None
                    info = info["entries"][0]

                video_id = info.get("id", "unknown")
                video_title = info.get("title", "Unknown")
                print(f"[Choreographer] Downloading: {video_title}")

                # If it was a search, we need to download the specific video URL now, 
                # or just let yt-dlp handle the search URL if it does so automatically.
                # yt-dlp download([url]) with ytsearch1: works, but let's be safe and use the web_url if available
                # or just pass the original url if it was a search.
                # Actually, ydl.download([url]) works for ytsearch1: too.
                ydl.download([url])

                # Find the downloaded file
                audio_files = list(self.download_dir.glob("*.wav"))
                if audio_files:
                    # Sort by modification time, return most recent
                    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return str(audio_files[0])

        except Exception as e:
            print(f"[Choreographer] Download error: {e}")

        return None

    def get_video_info(self, url: str) -> Optional[dict]:
        """Get video info without downloading."""
        try:
            import yt_dlp
        except ImportError:
            return None

        ydl_opts = {"extract_flat": False, "quiet": True, "no_warnings": True}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title"),
                    "duration": info.get("duration"),
                    "id": info.get("id"),
                }
        except Exception:
            return None


class SongAnalyzer:
    """Analyze audio files using Librosa."""

    def analyze(self, audio_path: str) -> SongAnalysis:
        """Perform full Librosa analysis of audio file."""
        print(f"[Choreographer] Analyzing: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr

        # Tempo and beat detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Extend beats if detection stopped early
        if len(beat_times) > 1 and beat_times[-1] < duration - 1.0:
            avg_interval = np.mean(np.diff(beat_times[-10:]))
            extended = []
            current = beat_times[-1]
            while current + avg_interval < duration:
                current += avg_interval
                extended.append(current)
            if extended:
                beat_times = np.concatenate([beat_times, extended])

        print(f"[Choreographer] Detected {len(beat_times)} beats at {tempo_val:.1f} BPM")

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Onset Strength (Perceived percussive attack)
        # matches same hop_length=512 by default
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Calculate energy at each beat
        energy_per_beat = self._compute_beat_energy(beat_times, rms, duration, len(y), sr)

        # Assign energy levels to 8-beat blocks
        sequence_assignments = self._assign_sequences(energy_per_beat)

        return SongAnalysis(
            audio_path=audio_path,
            duration=duration,
            tempo=tempo_val,
            beat_times=beat_times,
            energy_per_beat=energy_per_beat,
            sequence_assignments=sequence_assignments,
            energy_envelope=rms,
            onset_envelope=onset_env,
            envelope_sr=sr,
            hop_length=512,  # Librosa default
        )

    def _compute_beat_energy(
        self,
        beat_times: np.ndarray,
        rms: np.ndarray,
        duration: float,
        n_samples: int,
        sr: int,
    ) -> np.ndarray:
        """Map RMS energy to each beat time."""
        # RMS frames to time
        hop_length = 512  # Librosa default
        rms_times = np.arange(len(rms)) * hop_length / sr

        # Interpolate RMS at beat times
        energy_at_beats = np.interp(beat_times, rms_times, rms)

        # Normalize using percentiles (handles compression)
        p10 = np.percentile(energy_at_beats, 10)
        p90 = np.percentile(energy_at_beats, 90)
        if p90 - p10 > 0:
            normalized = np.clip((energy_at_beats - p10) / (p90 - p10), 0, 1)
        else:
            normalized = np.ones_like(energy_at_beats) * 0.5

        return normalized

    def _assign_sequences(self, energy_per_beat: np.ndarray) -> list[str]:
        """Assign "high"/"medium"/"low" energy level to each 8-beat block."""
        assignments = []
        n_beats = len(energy_per_beat)

        for i in range(0, n_beats, 8):
            block = energy_per_beat[i : i + 8]
            avg_energy = np.mean(block)

            if avg_energy >= 0.65:
                assignments.append("high")
            elif avg_energy <= 0.35:
                assignments.append("low")
            else:
                assignments.append("medium")

        return assignments


class Choreographer(DanceMode):
    """Mode B: Pre-analyzed YouTube video choreography."""

    MODE_ID = "B"
    MODE_NAME = "Choreographer"

    def __init__(self, safety_mixer: SafetyMixer, url: Optional[str] = None):
        super().__init__(safety_mixer)

        self.config = ChoreographerConfig()
        self.url = url
        self.analysis: Optional[SongAnalysis] = None

        # Load settings from mode_settings
        self._load_settings()

        # Threading
        self.stop_event = threading.Event()
        self.dance_thread: Optional[threading.Thread] = None
        self.audio_process: Optional[subprocess.Popen] = None

        # Sequence state
        self.current_sequence: list[dict] = []
        self.current_sequence_idx = 0
        self.last_sequence_used: dict[str, int] = {"high": -1, "medium": -1, "low": -1}

        # Current pose for interpolation
        self.current_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]

        # Breathing state
        self.breathing_time: float = 0.0

        # Status
        self._status = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "tempo": 0.0,
            "progress": 0.0,
            "current_beat": 0,
            "total_beats": 0,
            "energy_level": "",
            "is_breathing": False,
        }

    def set_url(self, url: str) -> None:
        """Set the YouTube URL for analysis."""
        self.url = url

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("B")
        self.config.amplitude_scale = settings.get("amplitude_scale", 0.5)
        self.config.interpolation_alpha = settings.get("interpolation_alpha", 0.3)
        self.config.antenna_energy_threshold = settings.get("antenna_energy_threshold", 0.5)

    def apply_settings(self, updates: dict[str, float]) -> None:
        """Apply settings updates (called from API for live tuning)."""
        if "amplitude_scale" in updates:
            self.config.amplitude_scale = updates["amplitude_scale"]
        if "interpolation_alpha" in updates:
            self.config.interpolation_alpha = updates["interpolation_alpha"]
        if "antenna_energy_threshold" in updates:
            self.config.antenna_energy_threshold = updates["antenna_energy_threshold"]

    async def start(self) -> None:
        """Start the choreographer."""
        if self.running:
            return

        if not self.url:
            print(f"[{self.MODE_NAME}] No URL provided")
            self._status["state"] = "error"
            return

        print(f"[{self.MODE_NAME}] Starting with URL: {self.url}")
        self._status["state"] = "downloading"

        # Download audio
        downloader = YouTubeDownloader(self.config.download_dir)
        audio_path = downloader.download_audio(self.url)

        if not audio_path:
            print(f"[{self.MODE_NAME}] Download failed")
            self._status["state"] = "error"
            return

        # Analyze audio
        self._status["state"] = "analyzing"
        analyzer = SongAnalyzer()
        self.analysis = analyzer.analyze(audio_path)

        # Calculate dynamic range stats for info logging only
        # (Normalization is now removed, we use raw values + sensitivity)
        rms = self.analysis.energy_envelope
        if len(rms) > 0:
            logger.info(f"[Choreographer] RMS Range: {np.min(rms):.4f} - {np.max(rms):.4f}")
            
        if self.analysis.onset_envelope is not None and len(self.analysis.onset_envelope) > 0:
            logger.info(f"[Choreographer] Onset Range: {np.min(self.analysis.onset_envelope):.4f} - {np.max(self.analysis.onset_envelope):.4f}")
            
        self._last_log_time = time.time()

        self._status["tempo"] = self.analysis.tempo
        self._status["total_beats"] = len(self.analysis.beat_times)

        # Start playback
        self.stop_event.clear()
        self.running = True

        self.dance_thread = threading.Thread(target=self._dance_loop, daemon=True)
        self.dance_thread.start()

        self._status["running"] = True
        self._status["state"] = "dancing"
        print(f"[{self.MODE_NAME}] Started - dancing to {self.analysis.tempo:.1f} BPM")

    async def stop(self) -> None:
        """Stop the choreographer."""
        if not self.running:
            return

        print(f"[{self.MODE_NAME}] Stopping...")
        self.running = False
        self.stop_event.set()

        # Stop audio first (terminates ffplay)
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=5.0)
            except Exception:
                try:
                    self.audio_process.kill()
                except Exception:
                    pass
            self.audio_process = None

        # Wait for thread with timeout
        if self.dance_thread and self.dance_thread.is_alive():
            self.dance_thread.join(timeout=2.0)
            if self.dance_thread.is_alive():
                print(f"[{self.MODE_NAME}] Warning: dance thread did not terminate")
        self.dance_thread = None

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

    def _select_sequence(self, energy_level: str) -> list[dict]:
        """Select an 8-beat sequence for the given energy level."""
        key = f"{energy_level}_energy"
        sequences = EIGHT_BEAT_SEQUENCES.get(key, EIGHT_BEAT_SEQUENCES["medium_energy"])

        # Cycle through sequences, avoiding immediate repeats
        last_idx = self.last_sequence_used[energy_level]
        next_idx = (last_idx + 1) % len(sequences)
        self.last_sequence_used[energy_level] = next_idx

        return sequences[next_idx]

    def _compute_breathing_pose(self, t: float) -> MovementIntent:
        """Compute breathing/idle pose for organic motion between choreographed moves."""
        # Y sway (side-to-side)
        y_offset = self.config.breathing_y_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_y_freq * t
        )

        # Head roll (gentle tilt)
        roll_offset = self.config.breathing_roll_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_roll_freq * t
        )

        return MovementIntent(
            position=self.config.neutral_pos + np.array([0.0, y_offset, 0.0]),
            orientation=self.config.neutral_eul + np.array([roll_offset, 0.0, 0.0]),
            antennas=np.array([-0.15, 0.15]),
        )

    def _coords_to_offset(self, coords: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Convert choreography coords to position/orientation OFFSETS.

        Coords are [x, y, z, roll, pitch, yaw] in (cm, cm, cm, deg, deg, deg).
        Returns (position_offset, orientation_offset) in (m, m, m, rad, rad, rad).

        These are deltas meant to be ADDED to a base pose (like breathing).
        """
        scale = self.config.amplitude_scale

        position_offset = np.array([
            coords[0] * 0.01 * scale,
            coords[1] * 0.01 * scale,
            coords[2] * 0.01 * scale,
        ])
        orientation_offset = np.array([
            np.radians(coords[3] * scale),
            np.radians(coords[4] * scale),
            np.radians(coords[5] * scale),
        ])

        return position_offset, orientation_offset

    def apply_settings(self, settings: dict[str, float]) -> None:
        """Apply live setting updates."""
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                import logging
                logging.getLogger("uvicorn").info(f"Updated Choreographer setting: {key} = {value}")

    def _add_offset_to_intent(
        self,
        base_intent: MovementIntent,
        pos_offset: np.ndarray,
        ori_offset: np.ndarray,
        offset_scale: float = 1.0,
    ) -> MovementIntent:
        """Add position/orientation offsets to a base intent.

        offset_scale controls how much of the offset is applied (0.0-1.0+).
        """
        return MovementIntent(
            position=base_intent.position + pos_offset * offset_scale,
            orientation=base_intent.orientation + ori_offset * offset_scale,
            antennas=base_intent.antennas,
        )

    def _get_continuous_antennas(self, current_time: float) -> np.ndarray:
        """
        Compute antenna position based on continuous energy envelope.
        Provides constant motion feel (100Hz loop) rather than discrete beats.
        """
        if not self.analysis or self.analysis.energy_envelope.size == 0:
            return np.array([-0.1, 0.1])

        # Map current time to envelope index
        # Time = index * hop_length / sr  =>  index = time * sr / hop_length
        idx = int(current_time * self.analysis.envelope_sr / self.analysis.hop_length)
        
        # Clamp index
        if idx < 0: 
            return np.array([-0.1, 0.1])
        if idx >= len(self.analysis.energy_envelope):
            # Check if we should hold last value or zero
            if idx > len(self.analysis.energy_envelope) + 10: # Way past end
                return np.array([-0.1, 0.1])
            idx = len(self.analysis.energy_envelope) - 1

        # Direct Gain Control
        raw_rms = self.analysis.energy_envelope[idx]
        
        # 0. Apply Noise Gate (Threshold)
        threshold = getattr(self.config, 'antenna_energy_threshold', 0.25)
        if raw_rms < threshold:
             signal = 0.0
        else:
             signal = (raw_rms - threshold)

        # 1. Apply Fixed Pre-Amp (Gain)
        gain = getattr(self.config, 'antenna_gain', 20.0)
        signal = signal * gain
        
        # 2. Apply User Sensitivity (Slider)
        scaled_signal = signal * self.config.antenna_sensitivity
        
        # 3. Map directly to Splay (Clamp to Max Amplitude)
        # splay = min(scaled_signal, max_travel)
        max_travel = self.config.antenna_amplitude
        splay = min(scaled_signal, max_travel)
        
        # Rest: -0.1 (Left), 0.1 (Right)
        # Splay pushes them outward
        rest = self.config.antenna_rest_position
        
        left = rest - splay
        right = -rest + splay
        
        # [DEBUG] logging
        # Only log every ~0.5s to avoid spam
        if getattr(self, '_last_debug_time', 0.0) + 0.5 < time.time():
            logging.getLogger("uvicorn").info(f"[ANTENNA DEBUG] RMS: {raw_rms:.4f} | Gate: {threshold:.2f} | Gain: {gain:.1f} | Splay: {splay:.2f}")
            self._last_debug_time = time.time()
        
        return np.array([left, right])

    def _dance_loop(self) -> None:
        """Main dance execution loop with additive choreography on breathing.

        Breathing is the continuous BASE motion.
        Choreography coordinates are OFFSETS added on top of breathing.
        Antennas move continuously driven by song energy envelope.
        """
        if not self.analysis:
            return

        beat_times = self.analysis.beat_times
        sequence_assignments = self.analysis.sequence_assignments

        # Start audio playback
        self._start_audio_playback()

        start_time = time.time()
        last_time = start_time
        current_block = -1
        beat_in_block = 0
        energy_level = "medium"

        # Reset breathing time
        self.breathing_time = 0.0

        # Current offset (interpolated toward target offset)
        current_offset = np.zeros(6)  # [x, y, z, roll, pitch, yaw]

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now
            elapsed = now - start_time

            # Update breathing time (continuous)
            self.breathing_time += dt

            # Find current beat
            current_beat = np.searchsorted(beat_times, elapsed) - 1
            current_beat = max(0, min(current_beat, len(beat_times) - 1))

            # Update status
            self._status["current_beat"] = current_beat
            self._status["progress"] = elapsed / self.analysis.duration

            # Check if song is done
            if elapsed >= self.analysis.duration:
                break

            # Determine which 8-beat block we're in
            block_idx = current_beat // 8

            if block_idx != current_block:
                # New 8-beat block - select sequence
                current_block = block_idx
                beat_in_block = 0

                if block_idx < len(sequence_assignments):
                    energy_level = sequence_assignments[block_idx]
                    self.current_sequence = self._select_sequence(energy_level)
                    self._status["energy_level"] = energy_level

            self._status["is_breathing"] = True

            # 1. Compute breathing base
            breathing_intent = self._compute_breathing_pose(self.breathing_time)

            # 2. Compute continuous energy-driven antennas (REPLACED logic)
            # Use elapsed time relative to song start to sample energy envelope
            antennas = self._get_continuous_antennas(elapsed)

            # Get current beat within block
            beat_in_block = current_beat % 8
            beat_in_block = min(beat_in_block, len(self.current_sequence) - 1)

            # 3. Compute choreography OFFSET and ADD it to breathing
            if self.current_sequence:
                move = self.current_sequence[beat_in_block]
                target_coords = move["coords"]

                # Interpolate current offset toward target offset (smoothing)
                target = np.array(target_coords)
                alpha = self.config.interpolation_alpha
                current_offset += (target - current_offset) * alpha

                # Convert offset to position/orientation deltas (amplitude_scale applied inside)
                pos_offset, ori_offset = self._coords_to_offset(current_offset.tolist())

                # Add offset to breathing base
                final_intent = self._add_offset_to_intent(breathing_intent, pos_offset, ori_offset)
                
                # Apply timeline-interpolated antennas
                final_intent = MovementIntent(
                    position=final_intent.position,
                    orientation=final_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)
            else:
                # No sequence - just breathe with timeline antennas
                final_intent = MovementIntent(
                    position=breathing_intent.position,
                    orientation=breathing_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)

            time.sleep(0.02)  # 50Hz update rate

    def _start_audio_playback(self) -> None:
        """Start audio playback using ffplay."""
        if not self.analysis:
            return

        try:
            # Use ffplay for audio playback (minimal, quiet)
            # Pre-convert to 16kHz mono for Reachy Mini's DAC
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel", "quiet",
                "-af", "aresample=16000,pan=mono|c0=c0",
                self.analysis.audio_path,
            ]
            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[{self.MODE_NAME}] Audio playback error: {e}")
