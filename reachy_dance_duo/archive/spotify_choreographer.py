"""
Mode S: Spotify Choreographer - Hybrid Spotify/YouTube dance mode.

1. User selects song from Spotify.
2. System searches YouTube for "Artist - Title".
3. Downloads audio via yt-dlp.
4. Performs Librosa analysis (same as Mode B).
5. Plays on Spotify and syncs robot moves to local analysis.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..behaviors.base import DanceMode
from ..behaviors.choreographer import Choreographer, SongAnalysis, SongAnalyzer, YouTubeDownloader
from ..core.safety_mixer import MovementIntent
from .client import SpotifyClient
from .. import mode_settings

if TYPE_CHECKING:
    from ..core.safety_mixer import SafetyMixer


class SpotifyChoreographer(Choreographer):
    """Mode S: Hybrid Spotify/YouTube choreography."""

    MODE_ID = "S"
    MODE_NAME = "Spotify Choreographer"

    def __init__(self, safety_mixer: SafetyMixer, spotify_client: SpotifyClient):
        # Initialize parent Choreographer (Mode B) logic
        # We don't pass URL here, we set it later via set_spotify_track
        super().__init__(safety_mixer, url=None)
        
        self.spotify = spotify_client
        self.spotify_track: Optional[dict] = None
        self.sync_offset: float = 0.0  # Seconds to adjust sync (latency compensation)
        self.use_local_audio: bool = True  # Fallback flag
        
        # Override status with Mode S specifics
        self._status["mode"] = self.MODE_ID
        self._status["spotify_connected"] = False
        self._status["track_info"] = None

        # Re-load settings for Mode S (parent loaded Mode B)
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from mode_settings module for Mode S."""
        settings = mode_settings.get_mode_settings("S")
        self.config.amplitude_scale = settings.get("amplitude_scale", 0.5)
        self.config.interpolation_alpha = settings.get("interpolation_alpha", 0.3)
        self.config.antenna_sensitivity = settings.get("antenna_sensitivity", 1.0)
        self.config.antenna_amplitude = settings.get("antenna_amplitude", 3.15)

    async def set_spotify_track(self, track_info: dict) -> None:
        """Set the Spotify track to dance to."""
        self.spotify_track = track_info
        self._status["track_info"] = {
            "name": track_info["name"],
            "artist": track_info["artists"][0]["name"],
            "uri": track_info["uri"],
            "image": track_info["album"]["images"][0]["url"] if track_info["album"]["images"] else None
        }
        
        # Construct search query for YouTube
        query = f"{self._status['track_info']['artist']} - {self._status['track_info']['name']} audio"
        print(f"[{self.MODE_NAME}] Selected Spotify track: {query}")
        
        # We'll use this query as the "URL" for the YouTubeDownloader
        # The downloader needs to be updated to handle search queries if it doesn't already,
        # or we resolve it here. 
        # For now, let's assume we need to resolve it to a URL first or pass "ytsearch1:..."
        self.url = f"ytsearch1:{query}"

    async def start(self) -> None:
        """Start the hybrid choreography."""
        if self.running:
            return

        if not self.spotify_track:
            print(f"[{self.MODE_NAME}] No Spotify track selected")
            self._status["state"] = "error"
            return

        print(f"[{self.MODE_NAME}] Starting hybrid flow...")
        self._status["state"] = "downloading"

        # 1. Download & Analyze (Re-using Mode B logic)
        # We use the parent class's logic but with our search query
        # Note: YouTubeDownloader needs to support "ytsearch1:" prefix which yt-dlp does.
        
        # Run download in thread to not block async loop
        downloader = YouTubeDownloader(self.config.download_dir)
        # This might block, but we are in async start. 
        # Ideally we should run this in an executor, but for now direct call:
        audio_path = downloader.download_audio(self.url)

        if not audio_path:
            print(f"[{self.MODE_NAME}] Download/Search failed for: {self.url}")
            self._status["state"] = "error"
            return

        # 2. Analyze
        self._status["state"] = "analyzing"
        analyzer = SongAnalyzer()
        self.analysis = analyzer.analyze(audio_path)
        
        self._status["tempo"] = self.analysis.tempo
        self._status["total_beats"] = len(self.analysis.beat_times)

        # 3. Check for available Spotify devices and start playback
        print(f"[{self.MODE_NAME}] Checking for Spotify devices...")
        self.use_local_audio = True  # Default to local, override if Spotify works
        
        try:
            devices_response = await self.spotify.get_devices()
            devices = devices_response.get("devices", [])
            
            if devices:
                active_device = next((d for d in devices if d.get("is_active")), devices[0])
                print(f"[{self.MODE_NAME}] Found device: {active_device['name']}")
                
                # Start playback on the device
                await self.spotify.start_playback(
                    uris=[self.spotify_track["uri"]], 
                    device_id=active_device["id"]
                )
                self.use_local_audio = False
                print(f"[{self.MODE_NAME}] Spotify playback started on {active_device['name']}")
            else:
                print(f"[{self.MODE_NAME}] No Spotify devices available - using local audio")
        except Exception as e:
            print(f"[{self.MODE_NAME}] Spotify playback failed: {e} - using local audio")
        
        self._status["spotify_playback_active"] = not self.use_local_audio
        self._status["using_local_audio"] = self.use_local_audio

        # 4. Start Dance Loop
        self.stop_event.clear()
        self.running = True
        
        # Choose dance loop based on audio source
        if self.use_local_audio:
            # Use parent class's dance loop with local ffplay audio
            self.dance_thread = threading.Thread(target=self._dance_loop, daemon=True)
            print(f"[{self.MODE_NAME}] Using local audio playback (ffplay)")
        else:
            # Use Spotify-synced dance loop
            self.dance_thread = threading.Thread(target=self._spotify_dance_loop, daemon=True)
            print(f"[{self.MODE_NAME}] Syncing to Spotify playback")
        
        self.dance_thread.start()

        self._status["running"] = True
        self._status["state"] = "dancing"
        print(f"[{self.MODE_NAME}] Started - dancing to {self.analysis.tempo:.1f} BPM")

    def _spotify_dance_loop(self) -> None:
        """Main dance execution loop synced to Spotify with additive choreography.

        Breathing is the continuous BASE motion.
        Choreography coordinates are OFFSETS added on top of breathing.
        Antennas drop on energetic beats and spring back up.
        """
        if not self.analysis:
            return

        beat_times = self.analysis.beat_times
        energy_per_beat = self.analysis.energy_per_beat
        sequence_assignments = self.analysis.sequence_assignments

        # Create a new event loop for this thread to call async Spotify methods
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        current_block = -1
        energy_level = "medium"

        # Polling state
        last_poll_time = 0
        spotify_position_ms = 0
        local_start_time = time.time()
        last_time = local_start_time
        is_playing = True

        # Reset breathing time
        self.breathing_time = 0.0

        # Current offset (interpolated toward target offset)
        current_offset = np.zeros(6)  # [x, y, z, roll, pitch, yaw]

        # Antenna beat tracking
        last_beat_idx = -1
        beat_start_time = local_start_time
        current_beat_energy = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            # Update breathing time (continuous)
            self.breathing_time += dt

            # Poll Spotify every 1s to resync (API rate limit friendly)
            if now - last_poll_time > 1.0:
                try:
                    state = loop.run_until_complete(self.spotify.get_playback_state())
                    if state and state.get("item"):
                        if state["item"]["uri"] != self.spotify_track["uri"]:
                            print(f"[{self.MODE_NAME}] Track changed externally")
                            break

                        is_playing = state["is_playing"]
                        if is_playing:
                            spotify_position_ms = state["progress_ms"]
                            local_start_time = now - (spotify_position_ms / 1000.0)

                            # Adjust for latency (approximate round trip)
                            # We can tune this via self.sync_offset
                            local_start_time += self.sync_offset

                    last_poll_time = now
                except Exception as e:
                    print(f"[{self.MODE_NAME}] Poll error: {e}")

            if not is_playing:
                time.sleep(0.1)
                continue

            # Calculate current time in song based on last sync
            elapsed = time.time() - local_start_time

            # --- Additive Choreography on Breathing ---

            # Find current beat
            current_beat = np.searchsorted(beat_times, elapsed) - 1
            current_beat = max(0, min(current_beat, len(beat_times) - 1))

            # Detect beat transition for antenna drops
            if current_beat != last_beat_idx:
                last_beat_idx = current_beat
                beat_start_time = now
                # Get energy for this beat
                if current_beat < len(energy_per_beat):
                    current_beat_energy = energy_per_beat[current_beat]
                else:
                    current_beat_energy = 0.5  # Default

            # Time since beat started (for antenna decay)
            time_since_beat = now - beat_start_time

            # Update status
            self._status["current_beat"] = current_beat
            self._status["progress"] = elapsed / self.analysis.duration

            # Check if song is done (with buffer)
            if elapsed >= self.analysis.duration + 2.0:
                break

            # Determine which 8-beat block we're in
            block_idx = current_beat // 8

            if block_idx != current_block:
                current_block = block_idx
                if block_idx < len(sequence_assignments):
                    energy_level = sequence_assignments[block_idx]
                    self.current_sequence = self._select_sequence(energy_level)
                    self._status["energy_level"] = energy_level

            self._status["is_breathing"] = True  # Always breathing now

            # 1. Compute breathing as the BASE motion (always present)
            breathing_intent = self._compute_breathing_pose(self.breathing_time)

            # 2. Compute beat-reactive antennas
            # Use continuous energy-driven antenna logic (Multi-Feature RMS*Onset)
            antennas = self._get_continuous_antennas(elapsed)

            # Get current beat within block
            beat_in_block = current_beat % 8
            # 3. Compute choreography OFFSET and ADD it to breathing
            if self.current_sequence:
                beat_in_block = min(beat_in_block, len(self.current_sequence) - 1)
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
                # Apply beat-reactive antennas
                final_intent = MovementIntent(
                    position=final_intent.position,
                    orientation=final_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)
            else:
                # No sequence - just breathe with beat antennas
                final_intent = MovementIntent(
                    position=breathing_intent.position,
                    orientation=breathing_intent.orientation,
                    antennas=antennas,
                )
                self.mixer.send_intent(final_intent)

            time.sleep(0.02)  # 50Hz

        loop.close()
        
        # When loop ends, ensure we stop
        if self.running:
            # We can't call async stop() from here easily, so we rely on the main thread
            # or just reset the mixer
            self.mixer.reset()
            self._status["running"] = False
            self._status["state"] = "idle"

    async def stop(self) -> None:
        """Stop the choreographer and audio playback."""
        print(f"[{self.MODE_NAME}] Stopping...")

        # Mark as stopped first
        was_running = self.running
        self.running = False
        self.stop_event.set()

        # Always try to stop audio - ffplay for local, Spotify API for streaming
        if self.use_local_audio:
            # Kill ffplay process (inherited from parent Choreographer)
            if self.audio_process:
                try:
                    self.audio_process.terminate()
                    self.audio_process.wait(timeout=2.0)
                    print(f"[{self.MODE_NAME}] Audio process terminated")
                except Exception:
                    try:
                        self.audio_process.kill()
                    except Exception:
                        pass
                self.audio_process = None
        else:
            # Pause Spotify playback
            try:
                await self.spotify.pause_playback()
                print(f"[{self.MODE_NAME}] Spotify playback paused")
            except Exception as e:
                print(f"[{self.MODE_NAME}] Failed to pause Spotify: {e}")

        if not was_running:
            # Graceful - already stopped, just reset state
            self.mixer.reset()
            self._status["running"] = False
            self._status["state"] = "idle"
            print(f"[{self.MODE_NAME}] Already stopped (graceful)")
            return

        # Wait for dance thread to finish
        if self.dance_thread and self.dance_thread.is_alive():
            self.dance_thread.join(timeout=2.0)
        self.dance_thread = None

        self.mixer.reset()
        self._status["running"] = False
        self._status["state"] = "idle"
        print(f"[{self.MODE_NAME}] Stopped")
