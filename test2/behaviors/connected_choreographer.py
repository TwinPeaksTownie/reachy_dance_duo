"""Connected Choreographer - Pre-analyzed audio dance mode.

Uses YouTube Music to search for tracks, downloads via yt-dlp, performs
offline Librosa analysis, and generates choreographed dance sequences
based on song energy.

This is the "Performer" - knows the song in advance, can anticipate
changes, and delivers rehearsed movements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import av
import librosa
import numpy as np

from .. import mode_settings
from ..core.safety_mixer import MovementIntent
from .base import DanceMode

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

    from ..core.safety_mixer import SafetyMixer
    from ..youtube_music.client import YouTubeMusicClient

logger = logging.getLogger(__name__)


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

CHEESY_MOVIE_QUOTES = [
    (
        "Step Up: Revolution",
        "A profound meditation on the futility of art in a capitalist hellscape where corporate greed consumes even the purest expressions of rhythmic rebellion.",
    ),
    (
        "Dirty Dancing: Havana Nights",
        "A tragic allegory for the inevitable decay of passion under the crushing weight of geopolitical conflict and the meaningless passage of time.",
    ),
    (
        "Honey 2",
        "A grim reminder that individual talent is ultimately meaningless in a society structured to exploit the dreams of the disenfranchised for fleeting entertainment.",
    ),
    (
        "StreetDance 3D",
        "An existential nightmare exploring the hollowness of spectacle, where depth is simulated but connection remains perpetually out of reach.",
    ),
    (
        "Bring It On: Fight to the Finish",
        "A harrowing depicition of the tribalistic nature of humanity, proving that even in organized sport, we are but wolves tearing at each other's throats.",
    ),
    (
        "Save the Last Dance 2",
        "A bleak examination of the fallacy of second chances, illustrating that past traumas are not overcome, but merely buried beneath new, equally fragile illusions.",
    ),
    (
        "Burlesque",
        "A garish display of desperate people clinging to the wreckage of a dying industry, singing their sorrows to an audience that will never truly know them.",
    ),
    (
        "Center Stage: Turn It Up",
        "A crushing realization that ambition is a poison, turning friends into rivals and the joy of movement into a sterile, competitive commodity.",
    ),
    (
        "High Strung",
        "A dissonant symphony of shattered expectations, where the harmony of music and dance only serves to highlight the discordant chaos of modern existence.",
    ),
    (
        "Make It Happen",
        "A cruel joke of a title for a story about the paralyzing fear of failure and the quiet desperation of settling for a life you never wanted.",
    ),
    (
        "Flashdance",
        "A solitary struggle against the industrial machine, where welding sparks are the only warmth in a cold, unfeeling world that demands your labor and compliant rhythm.",
    ),
    (
        "Footloose (2011)",
        "A futile rage against authority that ultimately reveals rebellion as a temporary phase before inevitable assimilation into the oppressive societal norm.",
    ),
    (
        "Battle of the Year",
        "A portrayal of international cooperation that dissolves into petty egoism, suggesting that unity is an impossible dream in a fractured, competitive world.",
    ),
    (
        "Coyote Ugly",
        "A stark look at the commodification of female agency, where dreams of songwriting are drowned in alcohol and the male gaze on a sticky bar top.",
    ),
    (
        "Magic Mike XXL",
        "A road trip into the void, where the performance of masculinity masks a deep, aching loneliness that no amount of glitter or gyrations can heal.",
    ),
    (
        "Pitch Perfect 2",
        "A cacophony of forced cheer masking the terror of obsolescence, as a group of aging performers desperately clings to relevance in a world moving on without them.",
    ),
    (
        "Fame (2009)",
        "A harsh lesson that fame is not a ladder to the stars, but a meat grinder that chews up the young and hopeful, leaving only broken spirits in its wake.",
    ),
    (
        "You Got Served",
        "A brutal treatise on the transactional nature of respect, where dignity is won or lost on a dance floor that cares nothing for the souls leaving their sweat upon it.",
    ),
    (
        "Work It",
        "An ironic celebration of mediocrity, suggesting that in a world devoid of true merit, faking it until you make it is the only survival strategy left.",
    ),
    (
        "Feel the Beat",
        "A depressing saga of a fallen star forced to return to the mediocrity she escaped, finding not redemption, but the suffocating embrace of small-town stagnation.",
    ),
]


@dataclass
class SongAnalysis:
    """Results of Librosa song analysis."""

    audio_path: str
    duration: float
    tempo: float
    beat_times: np.ndarray[Any, np.dtype[np.float64]]
    energy_per_beat: np.ndarray[Any, np.dtype[np.float32]]  # 0-1 energy at each beat
    sequence_assignments: list[str]  # "high"/"medium"/"low" per 8-beat block
    # Continuous signal envelopes
    energy_envelope: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    onset_envelope: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    envelope_sr: int = 22050  # Sample rate of the envelope
    hop_length: int = 512


@dataclass
class ConnectedChoreographerConfig:
    """Configuration for Connected Choreographer mode."""

    download_dir: str = "downloads"
    output_dir: str = "output"
    amplitude_scale: float = 0.5  # Global movement scale
    interpolation_alpha: float = 0.3  # Smoothing between beat poses

    # Antenna Control
    antenna_sensitivity: float = 1.0  # Multiplier for antenna responsiveness
    antenna_amplitude: float = 3.15  # Max travel for antenna movement
    antenna_gain: float = 20.0  # Fixed pre-amplification for raw RMS signal
    antenna_energy_threshold: float = 0.25  # Energy threshold to trigger movement

    # Breathing motion (between choreographed movements)
    breathing_y_amplitude: float = 0.016  # Y sway amplitude (meters)
    breathing_y_freq: float = 0.2  # Y sway frequency (Hz)
    breathing_roll_amplitude: float = 0.222  # Roll amplitude (radians)
    breathing_roll_freq: float = 0.15  # Roll frequency (Hz)

    # Antenna beat drops
    antenna_rest_position: float = -0.1  # Resting position (near top)
    antenna_drop_max: float = 2.0  # Maximum drop magnitude
    antenna_decay_rate: float = 4.0  # How fast antennas spring back

    # Neutral pose for breathing reference
    neutral_pos: np.ndarray[Any, np.dtype[np.float64]] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.01])
    )
    neutral_eul: np.ndarray[Any, np.dtype[np.float64]] = field(
        default_factory=lambda: np.zeros(3)
    )


class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    def __init__(
        self,
        download_dir: str = "downloads",
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the YouTubeDownloader.

        Args:
            download_dir: Directory to save downloaded files.
            log_callback: Callback function for logging download progress.

        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.log = log_callback if log_callback else lambda x: None

    def download_audio(self, url: str) -> Optional[str]:
        """Download audio from YouTube URL or search query.

        Args:
            url: Full YouTube URL or search query prefixed with "ytsearch1:".

        Returns:
            Path to downloaded WAV file or None if failed.

        """
        # Reconfigure stdout/stderr to use UTF-8 with error replacement
        try:
            if hasattr(sys.stdout, "reconfigure") and sys.stdout is not None:
                getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure") and sys.stderr is not None:
                getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")
        except Exception:
            pass  # If reconfigure fails, continue with original encoding

        # Ensure homebrew bins are in PATH for JS runtime (deno/node) detection on Mac
        if sys.platform == "darwin":
            # Add common homebrew paths if they exist
            extra_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
            current_path = os.environ.get("PATH", "")
            for p in extra_paths:
                if p not in current_path and os.path.isdir(p):
                    current_path = f"{p}:{current_path}"
            os.environ["PATH"] = current_path

        try:
            import yt_dlp  # type: ignore
        except ImportError:
            logger.error("[ConnectedChoreographer] yt-dlp not installed")
            return None

        output_template = str(self.download_dir / "%(title)s [%(id)s].%(ext)s")

        # Custom logger class to handle encoding issues
        class UTF8Logger:
            def debug(self, msg: str) -> None:
                if msg.startswith("[debug] "):
                    pass
                else:
                    self.info(msg)

            def info(self, msg: str) -> None:
                try:
                    logger.info(f"[yt-dlp] {msg}")
                except UnicodeEncodeError:
                    logger.info(
                        f"[yt-dlp] {msg.encode('utf-8', errors='replace').decode('utf-8')}"
                    )

            def warning(self, msg: str) -> None:
                try:
                    logger.warning(f"[yt-dlp] {msg}")
                except UnicodeEncodeError:
                    logger.warning(
                        f"[yt-dlp] {msg.encode('utf-8', errors='replace').decode('utf-8')}"
                    )

            def error(self, msg: str) -> None:
                try:
                    logger.error(f"[yt-dlp] {msg}")
                except UnicodeEncodeError:
                    logger.error(
                        f"[yt-dlp] {msg.encode('utf-8', errors='replace').decode('utf-8')}"
                    )

        ydl_opts: dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "extract_flat": False,
            "noplaylist": True,  # Prevent downloading entire playlists
            "max_downloads": 5,  # Allow for info extraction + multiple format attempts
            "quiet": False,  # Enable output so our logger handles it
            "no_warnings": False,
            "noprogress": True,
            "logger": UTF8Logger(),
            # Fix for 403 Forbidden and JS runtime issues
            "js_runtimes": {r: {} for r in ["deno", "node"] if shutil.which(r)},
            "remote_components": "ejs:github",  # Solve JS challenges
            "allow_unplayable_formats": False,  # Changed back to False to avoid clutter
            # Fix encoding issues with Unicode characters in titles
            "restrictfilenames": True,  # Replace special chars with underscores
            "windowsfilenames": True,  # Additional filename sanitization
            # Legacy compat options for better encoding handling
            "compat_opts": ["no-live-chat"],
            "postprocessors": [],
            "postprocessor_args": {},
        }

        try:
            with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    logger.error(
                        f"[ConnectedChoreographer] Failed to extract info for: {url}"
                    )
                    return None

                # Handle search results (playlist)
                if "entries" in info:
                    entries = info["entries"]
                    if not entries or entries[0] is None:
                        logger.error(
                            f"[ConnectedChoreographer] No search results for: {url}"
                        )
                        return None
                    info = entries[0]

                video_title = str(info.get("title", "Unknown"))
                video_id = str(info.get("id", "Unknown"))
                # Use the canonical webpage URL for the actual download to avoid playlist params
                download_target = info.get("webpage_url", url)

                # Safely encode title for logging
                safe_title = video_title.encode("utf-8", errors="replace").decode(
                    "utf-8"
                )
                logger.info(f"[ConnectedChoreographer] Downloading: {safe_title}")
                self.log(f"Downloading: {safe_title}")

                ydl.download([download_target])

                # yt-dlp might download various formats without conversion
                # We search for any file containing the video ID
                audio_files = list(self.download_dir.glob(f"*[{video_id}]*"))
                audio_files = [
                    f
                    for f in audio_files
                    if f.suffix in [".wav", ".mp3", ".webm", ".m4a", ".flac", ".opus"]
                ]
                
                if not audio_files:
                     # Fallback to any recent audio file if specific ID match fails
                    audio_files = list(self.download_dir.glob("*"))
                    audio_files = [
                        f
                        for f in audio_files
                        if f.suffix in [".wav", ".mp3", ".webm", ".m4a", ".flac", ".opus"]
                    ]

                if audio_files:
                    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return str(audio_files[0])

        except UnicodeEncodeError as e:
            logger.error(f"[ConnectedChoreographer] Encoding error: {e}")
            self.log("Download failed: encoding error in video metadata")
        except Exception as e:
            # Safely format error message
            try:
                error_msg = str(e)
            except UnicodeEncodeError:
                error_msg = str(e).encode("utf-8", errors="replace").decode("utf-8")
            logger.error(f"[ConnectedChoreographer] Download error: {error_msg}")
            self.log(f"Download Error: {error_msg}")
        return None


def load_audio_av(
    file_path: str, target_sr: int = 11025
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
    """Load audio file using PyAV and resample to target_sr.

    Args:
        file_path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        Tuple of (audio_data, sample_rate).
        audio_data is a 1D float32 numpy array (mono).

    """
    try:
        container = av.open(file_path)
        stream = container.streams.audio[0]

        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=target_sr,
        )

        frames = []
        for frame in container.decode(stream):
            frame.pts = None  # Reset pts to avoid issues with non-monotonic pts
            frames.extend(resampler.resample(frame))

        # Handle any remaining buffered frames
        # (Though resampler.resample shouldn't have much buffer if any?)

        all_samples = []
        for frame in frames:
            # We requested 'flt' (float32), mono
            # PyAV frame.to_ndarray() returns (n_channels, n_samples)
            # For mono, it's (1, n_samples)
            arr = frame.to_ndarray()
            all_samples.append(arr[0])

        if not all_samples:
            return np.array([], dtype=np.float32), target_sr

        y = np.concatenate(all_samples)
        return y.astype(np.float32), target_sr

    except Exception as e:
        logger.error(f"[ConnectedChoreographer] PyAV load error: {e}")
        raise RuntimeError(f"Failed to load audio with PyAV: {e}")


class SongAnalyzer:
    """Analyze audio files using Librosa."""

    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        """Initialize the SongAnalyzer.

        Args:
            log_callback: Callback function for logging analysis progress.

        """
        self.log = log_callback if log_callback else lambda x: None

    def analyze(self, audio_path: str) -> SongAnalysis:
        """Perform full Librosa analysis of audio file.

        Optimized for Raspberry Pi - uses lower sample rate for faster processing.
        """
        logger.info(f"[ConnectedChoreographer] Analyzing: {audio_path}")
        self.log(f"Analyzing audio: {Path(audio_path).name}")

        # Load audio using PyAV to avoid system ffmpeg dependency
        analysis_sr = 11025
        try:
            y, sr = load_audio_av(audio_path, target_sr=analysis_sr)
        except Exception as e:
            self.log(f"Audio load warning: {e}")
            # Fallback or re-raise
            raise

        # sr is int from load_audio_av
        duration = len(y) / sr

        self.log(f"Audio loaded: {duration:.1f}s at {sr}Hz")

        # Tempo and beat detection
        self.log("Detecting tempo and beats...")
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Extend beats if detection stopped early
        if len(beat_times) > 1 and beat_times[-1] < duration - 1.0:
            avg_interval = float(np.mean(np.diff(beat_times[-10:])))
            extended = []
            current = float(beat_times[-1])
            while current + avg_interval < duration:
                current += avg_interval
                extended.append(current)
            if extended:
                beat_times = np.concatenate([beat_times, np.array(extended)])

        logger.info(
            f"[ConnectedChoreographer] Detected {len(beat_times)} beats at {tempo_val:.1f} BPM"
        )
        self.log(f"Detected {len(beat_times)} beats at {tempo_val:.1f} BPM")

        # RMS energy (using smaller hop for faster computation)
        self.log("Computing energy envelope...")
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Onset Strength
        self.log("Computing onset envelope...")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Calculate energy at each beat
        self.log("Mapping energy to beats...")
        energy_per_beat = self._compute_beat_energy(
            beat_times, rms, duration, len(y), sr
        )

        # Assign energy levels to 8-beat blocks
        sequence_assignments = self._assign_sequences(energy_per_beat)

        self.log(f"Analysis complete! {len(sequence_assignments)} sequences planned")

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
            hop_length=hop_length,
        )

    def _compute_beat_energy(
        self,
        beat_times: np.ndarray[Any, np.dtype[np.float64]],
        rms: np.ndarray[Any, np.dtype[np.float32]],
        duration: float,
        n_samples: int,
        sr: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Map RMS energy to each beat time."""
        hop_length = 512
        rms_times = np.arange(len(rms)) * hop_length / sr

        # Interpolate RMS at beat times
        energy_at_beats = np.interp(beat_times, rms_times, rms)

        # Normalize using percentiles
        p10 = np.percentile(energy_at_beats, 10)
        p90 = np.percentile(energy_at_beats, 90)
        if p90 - p10 > 0:
            normalized = np.clip((energy_at_beats - p10) / (p90 - p10), 0, 1)
        else:
            normalized = np.ones_like(energy_at_beats) * 0.5

        return np.asanyarray(normalized, dtype=np.float32)

    def _assign_sequences(
        self, energy_per_beat: np.ndarray[Any, np.dtype[np.float32]]
    ) -> list[str]:
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


class ConnectedChoreographer(DanceMode):
    """Beat Bandit - Pre-analyzed audio dance mode.

    Uses YouTube Music for track search and selection.
    """

    MODE_ID = "beat_bandit"
    MODE_NAME = "Beat Bandit"

    # Asymmetric physics for body_yaw (hip sway) - similar to synthwave_serenade
    BODY_YAW_PHYSICS = {"attack": 0.25, "decay": 0.15}

    def __init__(
        self,
        safety_mixer: SafetyMixer,
        mini: ReachyMini,
        ytmusic_client: Optional[YouTubeMusicClient] = None,
    ):
        """Initialize the ConnectedChoreographer behavior.

        Args:
            safety_mixer: The safety mixer to use for movement.
            mini: ReachyMini instance for audio control.
            ytmusic_client: Optional YouTube Music client for track search.

        """
        super().__init__(safety_mixer)

        self.mini = mini
        self.config = ConnectedChoreographerConfig()
        self.ytmusic = ytmusic_client
        self.analysis: Optional[SongAnalysis] = None

        # Audio source (set via set_ytmusic_track)
        self.youtube_url: Optional[str] = None
        self.ytmusic_track: Optional[dict[str, Any]] = None

        # Load settings
        self._load_settings()

        # Threading
        self.stop_event = threading.Event()
        self.dance_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_process: Optional[subprocess.Popen[Any]] = None
        self.prep_task: Optional[asyncio.Task[None]] = None
        
        self.audio_playback_data: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None

        # Sequence state
        self.current_sequence: list[dict[str, Any]] = []
        self.current_sequence_idx = 0
        self.last_sequence_used: dict[str, int] = {"high": -1, "medium": -1, "low": -1}

        # Current pose for interpolation
        self.current_pose: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(6)

        # Body yaw state (separate from pose interpolation for snappy hip sway)
        self._current_body_yaw = 0.0

        # Breathing state
        self.breathing_time: float = 0.0

        # Status
        self._status: dict[str, Any] = {
            "mode": self.MODE_ID,
            "running": False,
            "state": "idle",
            "tempo": 0.0,
            "progress": 0.0,
            "current_beat": 0,
            "total_beats": 0,
            "energy_level": "",
            "is_breathing": False,
            "source": None,  # "ytmusic"
            "track_info": None,
            "logs": [],
        }

    def _log(self, message: str) -> None:
        """Add a log message to the status."""
        logger.info(f"[{self.MODE_NAME}] {message}")
        timestamp = time.strftime("%H:%M:%S")
        self._status["logs"].append(f"[{timestamp}] {message}")
        # Keep only last 50 logs
        if len(self._status["logs"]) > 50:
            self._status["logs"] = self._status["logs"][-50:]

    def _load_settings(self) -> None:
        """Load settings from mode_settings module."""
        settings = mode_settings.get_mode_settings("beat_bandit")
        self.config.amplitude_scale = settings.get("amplitude_scale", 0.5)
        self.config.interpolation_alpha = settings.get("interpolation_alpha", 0.3)
        self.config.antenna_sensitivity = settings.get("antenna_sensitivity", 1.0)
        self.config.antenna_amplitude = settings.get("antenna_amplitude", 3.15)
        self.config.antenna_energy_threshold = settings.get(
            "antenna_energy_threshold", 0.25
        )
        self.config.antenna_gain = settings.get("antenna_gain", 20.0)

    def apply_settings(self, settings: dict[str, float]) -> None:
        """Apply live setting updates."""
        for key, value in settings.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated ConnectedChoreographer setting: {key} = {value}")

    async def set_ytmusic_track(self, track_info: dict[str, Any]) -> None:
        """Set YouTube Music track as audio source.

        Args:
            track_info: Dict with videoId, title, artists, thumbnails, album

        """
        self.ytmusic_track = track_info

        video_id = track_info.get("videoId")
        title = track_info.get("title", "Unknown")
        artists = track_info.get("artists", [])
        artist_name = artists[0].get("name", "Unknown") if artists else "Unknown"

        # Use regular YouTube URL instead of music.youtube.com for better yt-dlp compatibility
        self.youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        thumbnails = track_info.get("thumbnails", [])
        thumbnail_url = thumbnails[0].get("url") if thumbnails else None

        self._status["source"] = "ytmusic"
        self._status["track_info"] = {
            "title": title,
            "artist": artist_name,
            "videoId": video_id,
            "thumbnail": thumbnail_url,
        }

        self._log(f"Set YouTube Music track: {artist_name} - {title}")

    async def set_youtube_url(self, url: str) -> None:
        """Set direct YouTube URL as audio source.

        Args:
            url: YouTube URL (can be youtube.com or music.youtube.com)

        """
        # Normalize to regular youtube.com URL for better yt-dlp compatibility
        if "music.youtube.com" in url:
            url = url.replace("music.youtube.com", "youtube.com")

        # Remove playlist parameters to avoid "cascade" of downloads if a Mix/Playlist URL is pasted
        if "watch?v=" in url and "&list=" in url:
            url = url.split("&list=")[0]
        elif "playlist?list=" in url:
            # If it's a pure playlist URL, yt-dlp with noplaylist will pick the first video
            pass

        self.youtube_url = url
        self.ytmusic_track = None

        self._status["source"] = "youtube"
        self._status["track_info"] = {
            "title": "YouTube Video",
            "artist": "Direct URL",
            "url": url,
            "thumbnail": None,
        }

        self._log(f"Set YouTube URL: {url}")

    def _get_download_url(self) -> Optional[str]:
        """Get the URL to download audio from."""
        return self.youtube_url

    async def start(self) -> None:
        """Start the choreographer (non-blocking)."""
        if self.running:
            return

        download_url = self._get_download_url()
        if not download_url:
            logger.info(
                f"[{self.MODE_NAME}] Started without audio source - waiting for selection"
            )
            self._log("Waiting for audio selection...")
            return

        # Initialize status immediately
        self.running = True
        self._status["running"] = True
        self._status["state"] = "preparing"
        logger.info(f"[{self.MODE_NAME}] Starting preparation for: {download_url}")

        # Start background preparation task
        self.prep_task = asyncio.create_task(self._prepare_and_start(download_url))

    def _audio_streaming_loop(self) -> None:
        """Stream audio data to the robot in chunks."""
        if self.audio_playback_data is None:
            return
            
        logger.info(f"[{self.MODE_NAME}] Starting audio stream thread")
        
        # Audio settings
        sr = 48000
        chunk_time = 0.1  # 100ms
        chunk_size = int(sr * chunk_time)
        total_samples = len(self.audio_playback_data)
        
        idx = 0
        
        # Ensure media player is started
        try:
            self.mini.media.start_playing()
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Failed to start media player: {e}")
            return
            
        start_time = time.time()
        
        # Simple streaming loop
        while not self.stop_event.is_set() and idx < total_samples:
            iter_start = time.time()
            
            # Extract chunk
            end = min(idx + chunk_size, total_samples)
            chunk = self.audio_playback_data[idx:end]
            
            # Push to robot
            try:
                self.mini.media.push_audio_sample(chunk)
            except Exception as e:
                logger.error(f"[{self.MODE_NAME}] Push audio error: {e}")
                break
                
            idx = end
            
            # Precision sleep to maintain timing
            elapsed = time.time() - iter_start
            sleep_time = max(0.0, chunk_time - elapsed)
            time.sleep(sleep_time)
            
        logger.info(f"[{self.MODE_NAME}] Audio stream finished")

    def _start_audio_playback(self) -> None:
        """Start audio playback using manual streaming to bypass SDK bug."""
        if self.audio_playback_data is None:
            return

        self.audio_thread = threading.Thread(target=self._audio_streaming_loop, daemon=True)
        self.audio_thread.start()

    async def _prepare_and_start(self, download_url: str) -> None:
        """Background task to download, analyze, and start dancing."""
        try:
            # 1. Download
            self._log("Retrieving audio from YouTube Music...")

            # Run blocking download in executor
            loop = asyncio.get_running_loop()
            downloader = YouTubeDownloader(
                self.config.download_dir, log_callback=self._log
            )
            audio_path = await loop.run_in_executor(
                None, downloader.download_audio, download_url
            )

            if not audio_path:
                logger.error(f"[{self.MODE_NAME}] Download failed")
                self._log("Download failed")
                self._status["state"] = "error"
                self.running = False
                self._status["running"] = False
                return

            self._log("Audio Received")

            # 2. Load Audio for Playback (48kHz)
            self._log("Loading audio...")
            
            # Load at 48kHz native for ReSpeaker
            self.audio_playback_data, _ = await loop.run_in_executor(
                    None, lambda: load_audio_av(audio_path, target_sr=48000)
            )
            
            if len(self.audio_playback_data) == 0:
                 logger.error(f"[{self.MODE_NAME}] Low-level audio load failed")
                 self._status["state"] = "error"
                 self.running = False
                 return

            # 3. Analyze (Downsample for speed)
            self._status["state"] = "analyzing"
            self._log("Analyzing Beats...")
            
            # We can analyze the high-res data, but let's downsample for librosa speed
            # Simple decimation (approximate)
            analysis_sr = 12000 # 48000 / 4
            analysis_audio = self.audio_playback_data[::4]
            
            # Create a dummy analysis object to reuse logic or modify Analyzer
            # Actually, SongAnalyzer takes a path. We should modify it to take data or just save a temp small wav?
            # Or just pass the path and let it load again (it's fast enough).
            # Let's let Analyzer load from file separately to avoid refactoring Analyzer.
            
            analyzer = SongAnalyzer(log_callback=self._log)
            self.analysis = await loop.run_in_executor(
                None, analyzer.analyze, audio_path
            )
            assert self.analysis is not None

            # Log envelope stats
            if len(self.analysis.energy_envelope) > 0:
                rms = self.analysis.energy_envelope
                logger.info(
                    f"[{self.MODE_NAME}] RMS Range: {np.min(rms):.4f} - {np.max(rms):.4f}"
                )

            self._status["tempo"] = self.analysis.tempo
            self._status["total_beats"] = len(self.analysis.beat_times)

            # 4. Plan / Hype
            self._log("Planning moves that will blow your mind")
            await asyncio.sleep(1.5)

            # 5. Start Dance Thread
            if not self.running:
                return

            self.stop_event.clear()
            self.dance_thread = threading.Thread(target=self._dance_loop, daemon=True)
            self.dance_thread.start()

            self._status["state"] = "dancing"

            # Display Cheesy Movie Quote
            movie, desc = random.choice(CHEESY_MOVIE_QUOTES)
            self._log(f"Playing: {movie}")
            self._log(desc)

            logger.info(
                f"[{self.MODE_NAME}] Started - dancing to {self.analysis.tempo:.1f} BPM"
            )
            
            # Cleanup source file now that we have data in memory
            try:
                os.remove(audio_path)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Preparation failed: {e}")
            self._log(f"Error: {e}")
            self._status["state"] = "error"
            self.running = False
            self._status["running"] = False
            import traceback
            traceback.print_exc()

    async def stop(self) -> None:
        """Stop the choreographer."""
        if not self.running:
            return

        logger.info(f"[{self.MODE_NAME}] Stopping...")
        self.running = False
        self.stop_event.set()

        # Cancel prep task if running
        if self.prep_task and not self.prep_task.done():
            self.prep_task.cancel()
            try:
                await self.prep_task
            except asyncio.CancelledError:
                pass
            self.prep_task = None

        # Stop audio
        if hasattr(self, "mini") and hasattr(self.mini, "media"):
            try:
                self.mini.media.stop_playing()
            except Exception as e:
                logger.error(f"[{self.MODE_NAME}] Error stopping audio: {e}")
        self.audio_process = None

        # Wait for thread
        if self.dance_thread and self.dance_thread.is_alive():
            self.dance_thread.join(timeout=2.0)
        self.dance_thread = None
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        self.audio_thread = None

        # Return to neutral
        self.mixer.reset()

        # Cleanup downloaded file
        if (
            self.analysis
            and self.analysis.audio_path
            and os.path.exists(self.analysis.audio_path)
        ):
            try:
                os.remove(self.analysis.audio_path)
                self._log("Deleted temporary audio file")
            except Exception as e:
                logger.error(f"[{self.MODE_NAME}] Error deleting file: {e}")

        self._status["running"] = False
        self._status["state"] = "idle"
        self._log("Stopped")
        logger.info(f"[{self.MODE_NAME}] Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status with JSON-serializable values."""
        status = self._status.copy()
        for key, value in status.items():
            if isinstance(value, (np.integer, np.floating)):
                status[key] = value.item()
            elif hasattr(value, "item"):
                status[key] = value.item()
        return status

    def _select_sequence(self, energy_level: str) -> list[dict[str, Any]]:
        """Select an 8-beat sequence for the given energy level."""
        key = f"{energy_level}_energy"
        sequences = EIGHT_BEAT_SEQUENCES.get(key, EIGHT_BEAT_SEQUENCES["medium_energy"])

        last_idx = self.last_sequence_used[energy_level]
        next_idx = (last_idx + 1) % len(sequences)
        self.last_sequence_used[energy_level] = next_idx

        return sequences[next_idx]

    def _compute_breathing_pose(self, t: float) -> MovementIntent:
        """Compute breathing/idle pose for organic motion."""
        y_offset = self.config.breathing_y_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_y_freq * t
        )
        roll_offset = self.config.breathing_roll_amplitude * np.sin(
            2.0 * np.pi * self.config.breathing_roll_freq * t
        )

        return MovementIntent(
            position=self.config.neutral_pos + np.array([0.0, y_offset, 0.0]),
            orientation=self.config.neutral_eul + np.array([roll_offset, 0.0, 0.0]),
            antennas=np.array([-0.15, 0.15]),
        )

    def _coords_to_offset(
        self, coords: list[float]
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        float,
    ]:
        """Convert choreography coords to position/orientation offsets and body_yaw."""
        scale = self.config.amplitude_scale

        position_offset = np.array(
            [
                coords[0] * 0.01 * scale,
                coords[1] * 0.01 * scale,
                coords[2] * 0.01 * scale,
            ]
        )
        # Yaw now goes to body_yaw (hip sway), not head orientation
        orientation_offset = np.array(
            [
                np.radians(coords[3] * scale),  # roll
                np.radians(coords[4] * scale),  # pitch
                0.0,  # head yaw stays neutral
            ]
        )
        body_yaw = np.radians(coords[5] * scale)  # yaw -> hip sway

        return position_offset, orientation_offset, body_yaw

    def _add_offset_to_intent(
        self,
        base_intent: MovementIntent,
        pos_offset: np.ndarray[Any, np.dtype[np.float64]],
        ori_offset: np.ndarray[Any, np.dtype[np.float64]],
        body_yaw: float = 0.0,
    ) -> MovementIntent:
        """Add position/orientation offsets to a base intent."""
        return MovementIntent(
            position=base_intent.position + pos_offset,
            orientation=base_intent.orientation + ori_offset,
            antennas=base_intent.antennas,
            body_yaw=body_yaw,
        )

    def _get_continuous_antennas(
        self, current_time: float
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Compute antenna position based on continuous energy envelope."""
        if not self.analysis or self.analysis.energy_envelope.size == 0:
            return np.array([-0.1, 0.1])

        idx = int(current_time * self.analysis.envelope_sr / self.analysis.hop_length)

        if idx < 0:
            return np.array([-0.1, 0.1])
        if idx >= len(self.analysis.energy_envelope):
            if idx > len(self.analysis.energy_envelope) + 10:
                return np.array([-0.1, 0.1])
            idx = len(self.analysis.energy_envelope) - 1

        raw_rms = self.analysis.energy_envelope[idx]

        # Apply noise gate
        threshold = self.config.antenna_energy_threshold
        if raw_rms < threshold:
            signal = 0.0
        else:
            signal = raw_rms - threshold

        # Apply gain and sensitivity
        signal = signal * self.config.antenna_gain * self.config.antenna_sensitivity

        # Map to splay
        max_travel = self.config.antenna_amplitude
        splay = min(signal, max_travel)

        rest = self.config.antenna_rest_position
        left = rest - splay
        right = -rest + splay

        return np.array([left, right])

    def _dance_loop(self) -> None:
        """Execute the main dance loop."""
        if not self.analysis:
            return

        beat_times = self.analysis.beat_times
        sequence_assignments = self.analysis.sequence_assignments

        # Start audio playback
        self._start_audio_playback()
        self._dance_loop_local_audio(beat_times, sequence_assignments)

    def _convert_to_wav(self, input_path: str) -> Optional[str]:
        """Convert audio file to WAV using PyAV (decoding) and SoundFile (encoding).

        Forces conversion to Mono 48kHz (standard for ReSpeaker) to ensure
        compatibility and avoid channel mapping issues.
        """
        try:
            import soundfile as sf
            
            # Use PyAV to decode source (handles webm, m4a, etc.)
            container = av.open(input_path)
            stream = container.streams.audio[0]
            
            # Force Mono 48kHz for reliability
            resampler = av.AudioResampler(
                format="fltp", # Float planar (easiest to handle as numpy)
                layout="mono", 
                rate=48000,
            )

            all_samples = []
            for frame in container.decode(stream):
                frame.pts = None
                for resampled in resampler.resample(frame):
                    # Validated: fltp + mono returns shape (1, samples)
                    arr = resampled.to_ndarray()
                    if arr.shape[0] > 0:
                        all_samples.append(arr[0]) # Get the single channel data
                    
            if not all_samples:
                logger.error(f"[{self.MODE_NAME}] No audio samples decoded")
                return None
                
            y = np.concatenate(all_samples)
            
            # Check for silence (if mean amplitude is near zero)
            if np.mean(np.abs(y)) < 0.001:
                logger.warning(f"[{self.MODE_NAME}] Warning: Converted audio seems silent")
            
            # Output filename
            wav_path = str(Path(input_path).with_suffix(".wav"))
            
            # Write WAV (SoundFile handles 1D array as mono)
            sf.write(wav_path, y, 48000)
            
            size_mb = os.path.getsize(wav_path) / (1024 * 1024)
            logger.info(f"[{self.MODE_NAME}] Converted to WAV: {wav_path} ({size_mb:.2f} MB)")
            self._log(f"Audio ready ({size_mb:.1f}MB)")
            
            # Delete original if different
            if wav_path != input_path:
                try:
                   os.remove(input_path)
                except OSError:
                   pass
                   
            return wav_path
            
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] WAV conversion failed: {e}")
            self._log(f"Audio conversion failed: {e}")
            return None

    def _dance_loop_local_audio(
        self,
        beat_times: np.ndarray[Any, np.dtype[np.float64]],
        sequence_assignments: list[str],
    ) -> None:
        """Dance loop synced to local ffplay audio."""
        assert self.analysis is not None  # Guaranteed by _dance_loop check
        start_time = time.time()
        last_time = start_time
        current_block = -1
        energy_level = "medium"
        self.breathing_time = 0.0
        current_offset = np.zeros(6)

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now
            elapsed = now - start_time

            self.breathing_time += dt

            current_beat = int(np.searchsorted(beat_times, elapsed)) - 1
            current_beat = max(0, min(current_beat, len(beat_times) - 1))

            self._status["current_beat"] = current_beat
            self._status["progress"] = elapsed / self.analysis.duration

            if elapsed >= self.analysis.duration:
                break

            block_idx = int(current_beat // 8)

            if block_idx != current_block:
                current_block = block_idx
                if block_idx < len(sequence_assignments):
                    energy_level = sequence_assignments[block_idx]
                    self.current_sequence = self._select_sequence(energy_level)
                    self._status["energy_level"] = energy_level

            self._status["is_breathing"] = True

            breathing_intent = self._compute_breathing_pose(self.breathing_time)
            antennas = self._get_continuous_antennas(elapsed)

            beat_in_block = int(current_beat % 8)
            beat_in_block = (
                min(beat_in_block, len(self.current_sequence) - 1)
                if self.current_sequence
                else 0
            )

            if self.current_sequence:
                move = self.current_sequence[beat_in_block]
                target = np.array(move["coords"])
                alpha = self.config.interpolation_alpha
                current_offset += (target - current_offset) * alpha

                # Get target body_yaw directly from move (not through slow interpolation)
                target_yaw = np.radians(move["coords"][5] * self.config.amplitude_scale)

                # Apply asymmetric smoothing: fast attack, slower decay
                if abs(target_yaw) > abs(self._current_body_yaw):
                    yaw_alpha = self.BODY_YAW_PHYSICS["attack"]
                else:
                    yaw_alpha = self.BODY_YAW_PHYSICS["decay"]
                self._current_body_yaw += (
                    target_yaw - self._current_body_yaw
                ) * yaw_alpha

                pos_offset, ori_offset, _ = self._coords_to_offset(
                    current_offset.tolist()
                )
                final_intent = self._add_offset_to_intent(
                    breathing_intent, pos_offset, ori_offset, self._current_body_yaw
                )
                final_intent = MovementIntent(
                    position=final_intent.position,
                    orientation=final_intent.orientation,
                    antennas=antennas,
                    body_yaw=self._current_body_yaw,
                )
                self.mixer.send_intent(final_intent)
            else:
                final_intent = MovementIntent(
                    position=breathing_intent.position,
                    orientation=breathing_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)

            time.sleep(0.02)

    def _start_audio_playback(self) -> None:
        """Start audio playback using ReachyMini SDK."""
        if not self.analysis:
            return

        try:
            # play_sound relies on soundfile which needs WAV usually
            # We guaranteed WAV conversion in _prepare_and_start
            self.mini.media.play_sound(self.analysis.audio_path)
        except Exception as e:
            logger.error(f"[{self.MODE_NAME}] Audio playback error: {e}")
            self._log(f"Playback error: {e}")
