#!/usr/bin/env python3
"""Speech Analysis Module for Motion Generation.

Extracts audio and text features from TTS output to generate synchronized
robot motion timelines.

Based on reachy_mini_dancer's audio_analyzer.py but adapted for speech prosody.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import librosa
import librosa.display
import numpy as np


def load_audio_av(file_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file using PyAV to avoid libsndfile/soundfile dependency."""
    try:
        container = av.open(file_path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)

        all_samples = []
        for frame in container.decode(stream):
            frame.pts = None
            for resampled_frame in resampler.resample(frame):
                all_samples.append(resampled_frame.to_ndarray()[0])

        if not all_samples:
            return np.array([], dtype=np.float32), target_sr

        return np.concatenate(all_samples).astype(np.float32), target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {file_path} with PyAV: {e}")


from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SpeechAnalyzer:
    """Analyzes TTS audio and text for motion generation."""

    def __init__(self) -> None:
        """Initialize the speech analyzer."""
        self.sample_rate = 22050  # Librosa default, sufficient for speech
        self.hop_length = 512  # ~23ms between frames at 22050 Hz

    def analyze(self, audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze TTS audio file and optional text.

        Args:
            audio_path: Path to TTS audio file (.mp3, .wav)
            text: Original text that was synthesized (optional)

        Returns:
            Analysis dict with audio_features and text_features

        """
        logger.info(f"\n{'=' * 70}")
        logger.info("SPEECH ANALYSIS")
        logger.info(f"{'=' * 70}")
        logger.info(f"Audio: {audio_path}")
        if text:
            logger.info(
                f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}"
            )
        logger.info("")

        # Load audio using PyAV to avoid libsndfile dependency
        audio, sr = load_audio_av(audio_path, target_sr=self.sample_rate)

        # Calculate duration from samples and rate since we don't have librosa.load's auto duration
        duration = len(audio) / sr

        # Extract audio features
        audio_features = self._extract_audio_features(audio, duration)

        # Extract text features if provided
        text_features = {}
        if text:
            text_features = self._extract_text_features(text, duration)

        analysis = {
            "audio_file": audio_path,
            "text": text,
            "duration": float(duration),
            "sample_rate": self.sample_rate,
            "audio_features": audio_features,
            "text_features": text_features,
        }

        self._print_summary(analysis)

        return analysis

    def _extract_audio_features(
        self, audio: NDArray[np.float32], duration: float
    ) -> Dict[str, Any]:
        """Extract prosody features from audio."""
        # 1. Pitch contour (fundamental frequency over time)
        pitch_times, pitch_values, pitch_confidence = self._extract_pitch_contour(audio)

        # 2. Energy envelope (loudness over time)
        energy_times, energy_values = self._extract_energy_envelope(audio)

        # 3. Pause detection (silence regions between phrases)
        pauses = self._detect_pauses(audio)

        # 4. Speaking rate estimation
        speaking_rate = self._estimate_speaking_rate(audio, duration)

        # 5. Pitch statistics (for mood validation)
        pitch_stats = self._calculate_pitch_statistics(pitch_values, pitch_confidence)

        # 6. Energy statistics (for emphasis detection)
        energy_stats = self._calculate_energy_statistics(energy_values)

        return {
            "pitch_contour": {
                "times": pitch_times.tolist()
                if isinstance(pitch_times, np.ndarray)
                else pitch_times,
                "values": pitch_values.tolist()
                if isinstance(pitch_values, np.ndarray)
                else pitch_values,
                "confidence": pitch_confidence.tolist()
                if isinstance(pitch_confidence, np.ndarray)
                else pitch_confidence,
            },
            "pitch_stats": pitch_stats,
            "energy_envelope": {
                "times": energy_times.tolist()
                if isinstance(energy_times, np.ndarray)
                else energy_times,
                "values": energy_values.tolist()
                if isinstance(energy_values, np.ndarray)
                else energy_values,
            },
            "energy_stats": energy_stats,
            "pauses": pauses,
            "speaking_rate": float(speaking_rate),
        }

    def _extract_pitch_contour(
        self, audio: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Extract pitch (F0) over time using librosa's pyin."""
        # Use pyin (probabilistic YIN) for pitch tracking
        # fmin/fmax cover typical speech F0 range: 80-400 Hz
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=self.sample_rate,
            fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
            fmax=float(librosa.note_to_hz("C6")),  # ~1047 Hz (wider range for safety)
            hop_length=self.hop_length,
        )

        # Replace NaN values (unvoiced regions) with 0
        f0 = np.nan_to_num(f0, nan=0.0)

        # Generate time values for each pitch frame
        frame_duration = self.hop_length / self.sample_rate
        pitch_times = np.arange(len(f0)) * frame_duration

        # Use voiced probabilities as confidence
        pitch_confidence = voiced_probs

        return pitch_times, f0, pitch_confidence

    def _extract_energy_envelope(
        self, audio: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Extract RMS energy over time."""
        # Use librosa's RMS feature
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]

        # Generate time values
        frame_duration = self.hop_length / self.sample_rate
        energy_times = np.arange(len(rms)) * frame_duration

        return energy_times, rms

    def _detect_pauses(
        self,
        audio: NDArray[np.float32],
        threshold_db: float = -40.0,
        min_duration: float = 0.2,
    ) -> List[Dict[str, float]]:
        """Detect pauses (silence regions) in audio."""
        # Use librosa's split to find non-silent intervals
        # top_db is the threshold below reference to consider as silence
        intervals = librosa.effects.split(
            audio, top_db=abs(threshold_db), hop_length=self.hop_length
        )

        # Convert intervals to pauses (gaps between intervals)
        pauses = []

        for i in range(len(intervals) - 1):
            # End of current interval to start of next interval is a pause
            pause_start_sample = intervals[i][1]
            pause_end_sample = intervals[i + 1][0]

            pause_start_time = float(pause_start_sample / self.sample_rate)
            pause_end_time = float(pause_end_sample / self.sample_rate)
            pause_duration = pause_end_time - pause_start_time

            # Filter by minimum duration
            if pause_duration >= min_duration:
                pauses.append(
                    {
                        "start": pause_start_time,
                        "end": pause_end_time,
                        "duration": pause_duration,
                    }
                )

        return pauses

    def _estimate_speaking_rate(
        self, audio: NDArray[np.float32], duration: float
    ) -> float:
        """Estimate speaking rate (syllables per second) using onset detection."""
        # Use onset detection as proxy for syllable rate
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Calculate onset rate (onsets per second)
        num_onsets = len(onset_frames)
        rate = num_onsets / duration if duration > 0 else 0

        # Onset rate is a rough proxy for syllable rate in speech
        # Typical speech: 4-6 syllables/second
        return rate

    def _calculate_pitch_statistics(
        self, pitch_values: NDArray[np.float32], pitch_confidence: NDArray[np.float32]
    ) -> Dict[str, float]:
        """Calculate pitch statistics for mood validation."""
        # Filter out low-confidence and zero pitch values
        valid_mask = (pitch_confidence > 0.5) & (pitch_values > 0)
        valid_pitch = pitch_values[valid_mask]

        if len(valid_pitch) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "variance_coefficient": 0.0,
            }

        mean_pitch = float(np.mean(valid_pitch))
        std_pitch = float(np.std(valid_pitch))

        return {
            "mean": mean_pitch,
            "std": std_pitch,
            "min": float(np.min(valid_pitch)),
            "max": float(np.max(valid_pitch)),
            "range": float(np.max(valid_pitch) - np.min(valid_pitch)),
            "variance_coefficient": std_pitch / mean_pitch if mean_pitch > 0 else 0.0,
        }

    def _calculate_energy_statistics(
        self, energy_values: NDArray[np.float32]
    ) -> Dict[str, Any]:
        """Calculate energy statistics for emphasis detection."""
        from scipy.signal import find_peaks

        mean_energy = float(np.mean(energy_values))
        std_energy = float(np.std(energy_values))

        # Find peaks (potential emphasis points)
        # Peak = energy > mean + 0.2*std (lower threshold for more frequent nods)
        threshold = mean_energy + 0.2 * std_energy

        # Use find_peaks with minimum distance between peaks to prevent overlapping head nods
        # Nod duration = 400ms, so require 500ms spacing for clean separation
        frame_duration = self.hop_length / self.sample_rate
        min_distance_frames = int(0.5 / frame_duration)  # 0.5 seconds minimum spacing

        peak_indices, _ = find_peaks(
            energy_values, height=threshold, distance=min_distance_frames
        )

        # Convert frame indices to times
        peak_times = (peak_indices * frame_duration).tolist()

        return {
            "mean": mean_energy,
            "std": std_energy,
            "max": float(np.max(energy_values)),
            "dynamic_range": float(
                np.max(energy_values) / (np.min(energy_values) + 1e-10)
            ),
            "peak_times": peak_times,
            "peak_count": len(peak_times),
        }

    def _extract_text_features(self, text: str, duration: float) -> Dict[str, Any]:
        """Extract structural features from text."""
        # 1. Extract mood tag
        mood = self._extract_mood_tag(text)

        # 2. Detect questions
        questions = self._detect_questions(text)

        # 3. Detect emphasis (CAPS, **bold**)
        emphasis = self._detect_emphasis(text)

        # 4. Detect lists/enumeration
        enumerations = self._detect_enumerations(text)

        # 5. Estimate word timings (rough approximation)
        # Assume uniform speaking rate for now
        words = text.split()
        word_count = len(words)
        words_per_second = word_count / duration if duration > 0 else 0

        return {
            "mood": mood,
            "questions": questions,
            "emphasis": emphasis,
            "enumerations": enumerations,
            "word_count": word_count,
            "words_per_second": float(words_per_second),
        }

    def _extract_mood_tag(self, text: str) -> Optional[str]:
        """Extract <!-- MOOD: mood_name --> tag from text."""
        pattern = r"<!--\s*MOOD:\s*([a-zA-Z0-9_]+)\s*-->"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _detect_questions(self, text: str) -> List[Dict[str, Any]]:
        """Detect question sentences."""
        questions = []

        # Split into sentences (simple approach)
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for question mark or question words
            is_question = sentence.endswith("?") or any(
                sentence.lower().startswith(qw)
                for qw in [
                    "what",
                    "why",
                    "how",
                    "when",
                    "where",
                    "who",
                    "which",
                    "can",
                    "could",
                    "would",
                    "should",
                    "is",
                    "are",
                    "do",
                    "does",
                ]
            )

            if is_question:
                questions.append(
                    {
                        "text": sentence,
                        "has_question_mark": sentence.endswith("?"),
                    }
                )

        return questions

    def _detect_emphasis(self, text: str) -> List[Dict[str, str]]:
        """Detect emphasized words (CAPS, **bold**)."""
        emphasis = []

        # Detect CAPS words (at least 3 chars, all uppercase)
        caps_pattern = r"\b([A-Z]{3,})\b"
        for match in re.finditer(caps_pattern, text):
            emphasis.append(
                {
                    "word": match.group(1),
                    "type": "caps",
                }
            )

        # Detect **bold** markdown
        bold_pattern = r"\*\*([^*]+)\*\*"
        for match in re.finditer(bold_pattern, text):
            emphasis.append(
                {
                    "word": match.group(1),
                    "type": "bold",
                }
            )

        return emphasis

    def _detect_enumerations(self, text: str) -> List[Dict[str, Any]]:
        """Detect list items (first, second, third / 1. 2. 3.)."""
        enumerations = []

        # Ordinal numbers
        ordinal_pattern = (
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
        )
        for match in re.finditer(ordinal_pattern, text, re.IGNORECASE):
            enumerations.append(
                {
                    "text": match.group(1),
                    "type": "ordinal",
                }
            )

        # Numbered lists
        numbered_pattern = r"(\d+)\.\s+"
        for match in re.finditer(numbered_pattern, text):
            enumerations.append(
                {
                    "number": int(match.group(1)),
                    "type": "numbered",
                }
            )

        return enumerations

    def _print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print human-readable analysis summary."""
        audio_feat = analysis["audio_features"]
        text_feat = analysis.get("text_features", {})

        logger.info(f"Duration: {analysis['duration']:.2f}s")
        logger.info("")

        # Pitch
        pitch_stats = audio_feat["pitch_stats"]
        logger.info("🎵 PITCH CONTOUR:")
        logger.info(f"  Mean: {pitch_stats['mean']:.1f} Hz")
        logger.info(
            f"  Range: {pitch_stats['range']:.1f} Hz ({pitch_stats['min']:.1f} - {pitch_stats['max']:.1f})"
        )
        logger.info(
            f"  Variance: {pitch_stats['variance_coefficient']:.3f} (higher = more varied)"
        )
        logger.info("")

        # Energy
        energy_stats = audio_feat["energy_stats"]
        logger.info("🔊 ENERGY ENVELOPE:")
        logger.info(f"  Mean: {energy_stats['mean']:.4f}")
        logger.info(f"  Dynamic range: {energy_stats['dynamic_range']:.1f}x")
        logger.info(f"  Emphasis peaks: {energy_stats['peak_count']} detected")
        if energy_stats["peak_times"]:
            logger.info(
                f"  Peak times: {', '.join(f'{t:.2f}s' for t in energy_stats['peak_times'][:5])}"
            )
        logger.info("")

        # Pauses
        pauses = audio_feat["pauses"]
        logger.info(f"⏸️  PAUSES: {len(pauses)} detected")
        for i, pause in enumerate(pauses[:3]):  # Show first 3
            logger.info(
                f"  {i + 1}. {pause['start']:.2f}s - {pause['end']:.2f}s ({pause['duration']:.2f}s)"
            )
        if len(pauses) > 3:
            logger.info(f"  ... and {len(pauses) - 3} more")
        logger.info("")

        # Speaking rate
        logger.info(
            f"🗣️  SPEAKING RATE: {audio_feat['speaking_rate']:.1f} syllables/second"
        )
        logger.info("")

        # Text features
        if text_feat:
            if text_feat.get("mood"):
                logger.info(f"😊 MOOD: {text_feat['mood']}")
                logger.info("")

            questions = text_feat.get("questions", [])
            if questions:
                logger.info(f"❓ QUESTIONS: {len(questions)} detected")
                for q in questions:
                    logger.info(f'  - "{q["text"]}"')
                logger.info("")

            emphasis = text_feat.get("emphasis", [])
            if emphasis:
                logger.info(f"💪 EMPHASIS: {len(emphasis)} detected")
                for e in emphasis:
                    logger.info(f"  - {e['word']} ({e['type']})")
                logger.info("")

            enumerations = text_feat.get("enumerations", [])
            if enumerations:
                logger.info(f"🔢 ENUMERATIONS: {len(enumerations)} detected")
                for enum in enumerations:
                    if enum["type"] == "ordinal":
                        logger.info(f"  - {enum['text']}")
                    else:
                        logger.info(f"  - {enum['number']}.")
                logger.info("")

        logger.info(f"{'=' * 70}\n")


def main() -> None:
    """Test the speech analyzer on a sample TTS file."""
    # Example usage
    import sys

    if len(sys.argv) < 2:
        logger.info("Usage: python speech_analyzer.py <audio_file> [text]")
        logger.info("\nExample:")
        logger.info(
            '  python speech_analyzer.py sample.mp3 "I\'m not entirely sure about that."'
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else None

    analyzer = SpeechAnalyzer()
    analysis = analyzer.analyze(audio_path, text)

    # Optionally save to JSON
    output_path = Path(audio_path).with_suffix(".analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
