"""FastAPI application for Reachy Dance Suite.

Provides REST API and serves config UI for controlling dance modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, cast

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


from pydantic import BaseModel

from reachy_mini import ReachyMini

from . import mode_settings, move_config
from .behaviors.base import DanceMode
from .behaviors.connected_choreographer import ConnectedChoreographer
from .behaviors.live_groove import LiveGroove
from .config import APP_CONFIG, get_default_safety_config
from .core.motion_controller import MotionController
from .core.safety_mixer import SafetyMixer
from .youtube_music import YouTubeMusicClient

logger = logging.getLogger(__name__)


# Global state
class AppState:
    """Global application state."""

    mini: Optional[ReachyMini] = None
    safety_mixer: Optional[SafetyMixer] = None
    motion_controller: Optional[MotionController] = None
    ytmusic_client: Optional[YouTubeMusicClient] = None
    current_mode: Optional[DanceMode] = None
    modes: dict[str, type[DanceMode]] = {}
    external_mini: bool = False  # Flag: True if robot was injected externally


state = AppState()


def initialize_with_robot(mini: ReachyMini) -> None:
    """Initialize the app state with an externally-provided robot.

    Called by main.py when running as a ReachyMiniApp.
    """
    state.mini = mini
    state.external_mini = True

    # Initialize SafetyMixer
    safety_config = get_default_safety_config()
    state.safety_mixer = SafetyMixer(safety_config, state.mini)
    logger.info("[UltraDanceMix9000] SafetyMixer initialized")

    # Initialize MotionController (for face tracking, breathing, layered motion)
    # Note: Don't start the motion loop here - it starts when System Audio mode activates
    state.motion_controller = MotionController(state.mini, enabled=True)
    logger.info(
        "[UltraDanceMix9000] MotionController ready (starts with System Audio mode)"
    )

    # Initialize YouTube Music client (no auth needed for search)
    state.ytmusic_client = YouTubeMusicClient()
    logger.info(
        "[UltraDanceMix9000] YouTube Music Client initialized (unauthenticated)"
    )

    # Register available modes
    state.modes = {
        "live_groove": LiveGroove,
        "beat_bandit": ConnectedChoreographer,
    }
    logger.info(f"[UltraDanceMix9000] Registered modes: {list(state.modes.keys())}")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler - connects to robot on startup.

    If initialize_with_robot() was called first (external mode), skip connection.
    """
    # The robot connection is injected by ReachyMiniApp (daemon manager)
    if state.external_mini and state.mini is not None:
        yield
        return

    # Fallback for dev/testing (if ran without daemon manager)
    # This path is not used in production.
    yield


app = FastAPI(
    title="Reachy Dance Suite",
    description="Unified dance application for Reachy Mini",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for iOS app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static file path
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# API Endpoints
@app.get("/", response_model=None)
async def root() -> FileResponse | HTMLResponse:
    """Serve the config UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(
            content="<h1>Reachy Dance Suite</h1><p>Static files not found.</p>"
        )


# Pydantic models for API
class SafetyConfigUpdate(BaseModel):
    """Model for updating safety config."""

    z_threshold: Optional[float] = None
    max_pitch_at_low_z: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    intensity: Optional[float] = None


class ModeStartRequest(BaseModel):
    """Model for mode start request."""

    url: Optional[str] = None  # For Mode B
    profile_path: Optional[str] = (
        None  # For Mode A - "default" to load saved profile, path for custom
    )
    skip_calibration: bool = False  # For Mode A - skip calibration entirely
    force_calibration: bool = (
        False  # For Mode A - force fresh calibration (ignore default profile)
    )




@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Get current application status."""
    mode_status = None
    if state.current_mode:
        mode_status = state.current_mode.get_status()

    return {
        "connected": state.mini is not None,
        "current_mode": state.current_mode.MODE_ID if state.current_mode else None,
        "available_modes": list(state.modes.keys()),
        "mode_status": mode_status,
    }


@app.post("/api/mode/{mode_id}/start")
async def start_mode(
    mode_id: str, request: Optional[ModeStartRequest] = None
) -> dict[str, Any]:
    """Start a dance mode."""
    mode_id = mode_id.lower()

    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    if mode_id not in state.modes:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: {mode_id}. Available: {list(state.modes.keys())}",
        )

    # Stop current mode if running
    if state.current_mode and state.current_mode.running:
        await state.current_mode.stop()

    # Create and start new mode
    mode_class = state.modes[mode_id]

    # Live Groove accepts profile_path, skip_calibration, and force_calibration
    if mode_id == "live_groove":
        profile_path = request.profile_path if request else None
        skip_calibration = request.skip_calibration if request else False
        force_calibration = request.force_calibration if request else False
        # "default" means use the default profile path (let LiveGroove handle it)
        if profile_path == "default":
            profile_path = None  # LiveGroove will check DEFAULT_PROFILE_PATH

        # We know mode_class is LiveGroove here
        lg_class = cast(Any, mode_class)
        state.current_mode = lg_class(
            state.safety_mixer,
            state.mini,  # Pass robot for audio access
            profile_path=profile_path,
            skip_calibration=skip_calibration,
            force_calibration=force_calibration,
        )
    elif mode_id == "beat_bandit":
        # Connected Choreographer needs the track initialized first
        params = request.params if request else {}
        track_id = params.get("track_id")
        if not track_id:
            raise HTTPException(
                status_code=400, detail="track_id required for beat_bandit"
            )

        mode_instance = mode_class(state.safety_mixer, state.mini, state.ytmusic_client)
        success = await mode_instance.initialize_track(track_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize track")
        state.current_mode = mode_instance

    else:
        # Other modes use safety_mixer
        state.current_mode = mode_class(state.safety_mixer)

    if state.current_mode:
        await state.current_mode.start()

    return {
        "status": "started",
        "mode": mode_id,
        "mode_status": state.current_mode.get_status() if state.current_mode else None,
    }


@app.post("/api/mode/stop")
async def stop_mode() -> dict[str, Any]:
    """Stop the current dance mode."""
    if state.current_mode is None or not state.current_mode.running:
        return {"status": "already_stopped"}

    await state.current_mode.stop()

    return {
        "status": "stopped",
        "mode_status": state.current_mode.get_status() if state.current_mode else None,
    }


@app.get("/api/safety")
async def get_safety_config() -> dict[str, Any]:
    """Get current safety configuration."""
    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    config = state.safety_mixer.config
    return {
        "z_threshold": config.z_threshold,
        "max_pitch_at_low_z": config.max_pitch_at_low_z,
        "smoothing_alpha": config.smoothing_alpha,
        "intensity": config.intensity,
        "max_position": config.max_position.tolist(),
        "min_position": config.min_position.tolist(),
        "max_orientation": config.max_orientation.tolist(),
        "min_orientation": config.min_orientation.tolist(),
    }


@app.post("/api/safety")
async def update_safety_config(update: SafetyConfigUpdate) -> dict[str, Any]:
    """Update safety configuration (live tuning)."""
    if state.safety_mixer is None:
        raise HTTPException(status_code=503, detail="Robot not connected")

    # Apply updates
    updates = update.model_dump(exclude_none=True)
    state.safety_mixer.update_config(**updates)

    return {
        "status": "updated",
        "updated_fields": list(updates.keys()),
    }


# Move dampening endpoints
@app.get("/api/moves")
async def get_moves() -> dict[str, Any]:
    """Get all moves and their dampening values."""
    return {
        "moves": move_config.get_all_dampening(),
    }


@app.post("/api/moves")
async def update_moves(updates: dict[str, float]) -> dict[str, Any]:
    """Update dampening values for moves."""
    move_config.set_all_dampening(updates)
    return {
        "status": "updated",
        "moves": move_config.get_all_dampening(),
    }


@app.post("/api/moves/reset")
async def reset_moves() -> dict[str, Any]:
    """Reset all moves to default dampening values."""
    move_config.reset_to_defaults()
    return {
        "status": "reset",
        "moves": move_config.get_all_dampening(),
    }


# Move mirror endpoints
@app.get("/api/moves/mirror")
async def get_move_mirror() -> dict[str, Any]:
    """Get mirror settings for all mirrorable moves."""
    return {
        "mirrorable": move_config.get_mirrorable_moves(),
        "mirror": move_config.get_all_mirror(),
    }


@app.post("/api/moves/mirror")
async def update_move_mirror(updates: dict[str, bool]) -> dict[str, Any]:
    """Update mirror settings for moves."""
    move_config.set_all_mirror(updates)

    # If Live Groove is running, refresh its move list
    if state.current_mode and state.current_mode.MODE_ID == "live_groove":
        if hasattr(state.current_mode, "refresh_moves"):
            cast(Any, state.current_mode).refresh_moves()

    return {
        "status": "updated",
        "mirror": move_config.get_all_mirror(),
    }


@app.post("/api/moves/mirror/reset")
async def reset_move_mirror() -> dict[str, Any]:
    """Reset all mirror settings to defaults (all false)."""
    move_config.reset_mirror_to_defaults()

    # If Live Groove is running, refresh its move list
    if state.current_mode and state.current_mode.MODE_ID == "live_groove":
        if hasattr(state.current_mode, "refresh_moves"):
            cast(Any, state.current_mode).refresh_moves()

    return {
        "status": "reset",
        "mirror": move_config.get_all_mirror(),
    }


# Mode settings endpoints
@app.get("/api/mode-settings")
async def get_all_mode_settings() -> dict[str, Any]:
    """Get all mode settings."""
    return {
        "settings": mode_settings.get_all_settings(),
    }


@app.get("/api/mode-settings/{mode_id}")
async def get_mode_settings(mode_id: str) -> dict[str, Any]:
    """Get settings for a specific mode."""
    mode_id = mode_id.lower()  # Normalize to lowercase
    settings = mode_settings.get_mode_settings(mode_id)
    return {
        "mode": mode_id,
        "settings": settings,
    }


@app.post("/api/mode-settings/{mode_id}")
async def update_mode_settings(
    mode_id: str, updates: dict[str, float]
) -> dict[str, Any]:
    """Update settings for a mode and apply to running mode if active."""
    mode_id = mode_id.lower()  # Normalize to lowercase
    mode_settings.update_mode_settings(mode_id, updates)

    # If this mode is currently running, apply settings live
    if state.current_mode and state.current_mode.MODE_ID == mode_id:
        if hasattr(state.current_mode, "apply_settings"):
            cast(Any, state.current_mode).apply_settings(updates)

    return {
        "status": "updated",
        "mode": mode_id,
        "settings": mode_settings.get_mode_settings(mode_id),
    }


@app.post("/api/mode-settings/sync")
async def sync_mode_settings() -> dict[str, Any]:
    """Reload all settings from JSON file."""
    settings = mode_settings.sync_from_file()
    return {
        "status": "synced",
        "settings": settings,
    }


@app.post("/api/mode-settings/reset")
async def reset_mode_settings() -> dict[str, Any]:
    """Reset all mode settings to defaults."""
    settings = mode_settings.reset_to_defaults()
    return {
        "status": "reset",
        "settings": settings,
    }


# Profile Management Endpoints
PROFILE_PATH = Path(__file__).parent / "environment_profile.npz"


@app.get("/api/profile/status")
async def get_profile_status() -> dict[str, Any]:
    """Check if a saved environment profile exists."""
    if not PROFILE_PATH.exists():
        return {
            "exists": False,
            "path": str(PROFILE_PATH),
        }

    try:
        # Load metadata from profile
        data = np.load(PROFILE_PATH, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        return {
            "exists": True,
            "path": str(PROFILE_PATH),
            "created": metadata.get("created", "unknown"),
            "silence_rms": metadata.get("silence_rms", 0),
            "breathing_rms": metadata.get("breathing_rms", 0),
            "dance_rms": metadata.get("dance_rms", 0),
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(PROFILE_PATH),
            "error": str(e),
        }


@app.post("/api/profile/save")
async def save_profile() -> dict[str, Any]:
    """Save current calibration profile to disk."""
    if not state.current_mode or state.current_mode.MODE_ID != "live_groove":
        raise HTTPException(
            status_code=400, detail="Live Groove must be active to save profile"
        )

    mode = cast(Any, state.current_mode)

    # Check if calibration data exists
    if mode.silence_noise_profile is None or mode.breathing_noise_profile is None:
        raise HTTPException(
            status_code=400, detail="No calibration data - run calibration first"
        )

    try:
        # Compute RMS values for metadata
        silence_rms = float(np.sqrt(np.mean(mode.silence_noise_profile**2)))
        breathing_rms = float(np.sqrt(np.mean(mode.breathing_noise_profile**2)))
        dance_rms = (
            float(np.sqrt(np.mean(mode.dance_noise_profile**2)))
            if mode.dance_noise_profile is not None
            else 0.0
        )

        metadata = {
            "created": datetime.now().isoformat(),
            "silence_rms": silence_rms,
            "breathing_rms": breathing_rms,
            "dance_rms": dance_rms,
        }

        np.savez(
            PROFILE_PATH,
            silence_profile=mode.silence_noise_profile,
            breathing_profile=mode.breathing_noise_profile,
            dance_profile=mode.dance_noise_profile
            if mode.dance_noise_profile is not None
            else mode.breathing_noise_profile,
            metadata=json.dumps(metadata),
        )

        logger.info(f"[API] Saved environment profile to {PROFILE_PATH}")
        return {
            "status": "saved",
            "path": str(PROFILE_PATH),
            "created": metadata["created"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {e}")


# YouTube Music Endpoints
@app.get("/api/ytmusic/status")
async def get_ytmusic_status() -> dict[str, Any]:
    """Check if YouTube Music is available.

    No authentication required - search works without login.
    """
    if not state.ytmusic_client:
        return {"available": False, "message": "Client not initialized"}
    return {
        "available": state.ytmusic_client.is_available(),
        "message": "Ready to search (no login required)",
    }


@app.get("/api/ytmusic/search")
async def ytmusic_search(q: str) -> dict[str, Any]:
    """Search YouTube Music tracks.

    No authentication required - search works for everyone!
    """
    if not state.ytmusic_client:
        raise HTTPException(status_code=503, detail="YouTube Music not initialized")

    try:
        results = state.ytmusic_client.search(q)
        return {"tracks": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/mode/beat_bandit/track")
async def set_ytmusic_track(track: dict[str, Any]) -> dict[str, Any]:
    """Set the YouTube Music track for Connected Choreographer."""
    logger.info(f"[API] Received track selection: {track.get('title', 'Unknown')}")

    if not state.current_mode or state.current_mode.MODE_ID != "beat_bandit":
        logger.info("[API] Beat Bandit not active - activating now...")

        # Stop current mode if running
        if state.current_mode and state.current_mode.running:
            await state.current_mode.stop()

        # Initialize Beat Bandit
        if "beat_bandit" not in state.modes:
            raise HTTPException(
                status_code=500, detail="Beat Bandit mode not available"
            )

        mode_class = state.modes["beat_bandit"]
        # ConnectedChoreographer takes safety_mixer, mini, and ytmusic_client
        state.current_mode = mode_class(
            state.safety_mixer, state.mini, state.ytmusic_client
        )

    try:
        # If running (and was already active), stop playback so we can restart
        if state.current_mode.running:
            logger.info("[API] Stopping current playback...")
            await state.current_mode.stop()

        logger.info("[API] Setting track...")
        await cast(Any, state.current_mode).set_ytmusic_track(track)

        logger.info("[API] Starting playback...")
        await state.current_mode.start()

        logger.info(f"[API] Track set successfully: {track.get('title', 'Unknown')}")
        return {"status": "track_set", "track": track.get("title", "Unknown")}
    except Exception as e:
        logger.error(f"[API] Error setting track: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mode/beat_bandit/youtube")
async def set_youtube_url(request: dict[str, Any]) -> dict[str, Any]:
    """Set a direct YouTube URL for Connected Choreographer."""
    url = request.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    logger.info(f"[API] Received YouTube URL: {url}")

    if not state.current_mode or state.current_mode.MODE_ID != "beat_bandit":
        logger.info("[API] Beat Bandit not active - activating now...")

        # Stop current mode if running
        if state.current_mode and state.current_mode.running:
            await state.current_mode.stop()

        # Initialize Beat Bandit
        if "beat_bandit" not in state.modes:
            raise HTTPException(
                status_code=500, detail="Beat Bandit mode not available"
            )

        mode_class = state.modes["beat_bandit"]
        state.current_mode = mode_class(
            state.safety_mixer, state.mini, state.ytmusic_client
        )

    try:
        # If running, stop it first so we can restart with new URL
        if state.current_mode.running:
            logger.info("[API] Stopping current playback...")
            await state.current_mode.stop()

        logger.info("[API] Setting YouTube URL...")
        await cast(Any, state.current_mode).set_youtube_url(url)

        logger.info("[API] Starting playback...")
        await state.current_mode.start()

        logger.info(f"[API] YouTube URL set successfully: {url}")
        return {"status": "url_set", "url": url}
    except Exception as e:
        logger.error(f"[API] Error setting YouTube URL: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





# WebSocket for real-time status streaming
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time status updates."""
    await websocket.accept()

    try:
        while True:
            # Build status
            mode_status = None
            if state.current_mode:
                mode_status = state.current_mode.get_status()

            status = {
                "connected": state.mini is not None,
                "current_mode": state.current_mode.MODE_ID
                if state.current_mode
                else None,
                "available_modes": list(state.modes.keys()),
                "mode_status": mode_status,
            }

            await websocket.send_json(status)
            await asyncio.sleep(0.1)  # 10Hz updates

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create the app."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reachy_dance_suite.app:app",
        host=cast(str, APP_CONFIG["host"]),
        port=cast(int, APP_CONFIG["port"]),
        reload=False,
    )
