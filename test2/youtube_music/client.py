"""YouTube Music Client for Reachy Dance Suite.

Uses ytmusicapi for searching YouTube Music.
- Search works WITHOUT authentication (unauthenticated mode)
- This means zero setup required for users
- Scales to unlimited users with no API quotas
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ytmusicapi import YTMusic

logger = logging.getLogger(__name__)


class YouTubeMusicClient:
    """Client for YouTube Music API operations.

    Uses ytmusicapi in unauthenticated mode for searching.
    No login required - search just works out of the box.

    For downloading the actual audio, we use yt-dlp which also
    doesn't require authentication for public videos.
    """

    def __init__(self) -> None:
        """Initialize the YouTubeMusicClient."""
        self._ytmusic: Optional[YTMusic] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the YTMusic client (unauthenticated)."""
        try:
            from ytmusicapi import YTMusic

            # No auth = unauthenticated mode
            # This works for: search, get_song, get_artist, get_album, etc.
            self._ytmusic = YTMusic()
            logger.info("[YTMusic] Client initialized (unauthenticated mode)")
        except Exception as e:
            logger.error(f"[YTMusic] Failed to initialize: {e}")
            self._ytmusic = None

    def is_available(self) -> bool:
        """Check if the client is ready to use."""
        return self._ytmusic is not None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search YouTube Music for songs.

        Works without authentication - no login required!

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of track dicts with:
            - videoId: YouTube video ID (use with yt-dlp)
            - title: Song title
            - artists: List of artist dicts with 'name'
            - thumbnails: List of thumbnail dicts with 'url', 'width', 'height'
            - album: Album info dict (optional)
            - duration: Duration string (optional)

        """
        if not self._ytmusic:
            # Try to reinitialize
            self._initialize()
            if not self._ytmusic:
                raise RuntimeError("YouTube Music client not available")

        try:
            results = self._ytmusic.search(query, filter="songs", limit=limit)
        except Exception as e:
            logger.error(f"[YTMusic] Search failed: {e}")
            # Try reinitializing and retry once
            self._initialize()
            if self._ytmusic:
                results = self._ytmusic.search(query, filter="songs", limit=limit)
            else:
                raise

        # Normalize results to consistent format
        tracks = []
        for item in results:
            if item.get("resultType") != "song":
                continue

            track = {
                "videoId": item.get("videoId"),
                "title": item.get("title", "Unknown Title"),
                "artists": item.get("artists", []),
                "thumbnails": item.get("thumbnails", []),
                "album": item.get("album"),
                "duration": item.get("duration"),
                "duration_seconds": item.get("duration_seconds"),
            }
            tracks.append(track)

        return tracks

    def get_song_info(self, video_id: str) -> Dict[str, Any]:
        """Get detailed info for a specific song.

        Args:
            video_id: YouTube video ID

        Returns:
            Dict with song details

        """
        if not self._ytmusic:
            self._initialize()
            if not self._ytmusic:
                raise RuntimeError("YouTube Music client not available")

        return self._ytmusic.get_song(video_id)

    @staticmethod
    def get_video_url(video_id: str) -> str:
        """Get YouTube Music URL for a video ID."""
        return f"https://music.youtube.com/watch?v={video_id}"

    @staticmethod
    def get_youtube_url(video_id: str) -> str:
        """Get standard YouTube URL for a video ID (for yt-dlp)."""
        return f"https://www.youtube.com/watch?v={video_id}"
