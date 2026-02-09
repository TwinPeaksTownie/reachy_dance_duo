"""Reachy Ultra Dance Mix 9000 - Main Entry Point.

A comprehensive dance application for Reachy Mini with three modes:
- Live Groove: Real-time BPM-driven dancing from audio input
- Beat Bandit: YouTube integration with beat analysis
- Synthwave Serenade: Streaming audio control with face tracking

This wraps the dance suite as a ReachyMiniApp for the Reachy Mini dashboard.
"""

import asyncio
import logging
import socket
import threading
import time

import uvicorn

logger = logging.getLogger(__name__)

from reachy_mini import ReachyMini, ReachyMiniApp

from .app import app, initialize_with_robot, state


def _get_app_url(port: int = 9000) -> str:
    """Build the app URL based on the robot's hostname.

    Wireless Reachy Minis are accessed via mDNS (e.g. reachy-mini.local),
    so the dashboard tab needs to point there instead of localhost.
    Try the mDNS hostname first since wireless is the common case.
    """
    hostname = socket.gethostname()
    # Default to mDNS — wireless Reachy is the primary use case
    if hostname and hostname != "localhost":
        return f"http://{hostname}.local:{port}"
    return f"http://localhost:{port}"


class ReachyDanceDuo(ReachyMiniApp):
    """Reachy Dance Duo - Full dance suite for Reachy Mini.

    Features:
    - Live Groove: Dance to any music playing nearby using microphone input
    - Beat Bandit: Play YouTube videos with synchronized dancing
    - Synthwave Serenade: Stream audio for reactive movement and face tracking

    The app provides a web UI for configuration and control.
    """

    # App icon emoji (shown in dashboard)
    emoji: str = "🕺"
    # URL for the custom settings page (served by our FastAPI app)
    custom_app_url: str | None = _get_app_url()
    # Prevent daemon from starting its own basic server - we handle it ourselves
    dont_start_webserver: bool = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the dance suite.

        This starts the FastAPI server which provides:
        - Web UI for mode selection and configuration
        - REST API for controlling dance modes
        - WebSocket for real-time status updates
        """
        # Initialize app state with the provided robot
        initialize_with_robot(reachy_mini)

        # Debug: Print registered routes
        logger.info(f"[ReachyDanceDuo] Routes registered: {len(app.routes)}")
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "methods"):
                logger.info(f"  {route.methods} {route.path}")

        # Configure the server port
        port = 9000

        logger.info(
            f"[ReachyDanceDuo] App object id: {id(app)}, routes: {len(app.routes)}"
        )

        # Create uvicorn config - use the app object directly
        import logging

        logging.getLogger(
            "reachy_mini.daemon.backend.robot.backend.throttled"
        ).setLevel(logging.ERROR)

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",  # Enable info logging to see what's happening
        )

        # Create the server
        server = uvicorn.Server(config)

        # Run the server in a separate thread so we can monitor stop_event
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        logger.info(f"[ReachyDanceDuo] Started on http://0.0.0.0:{port}")

        # Main loop - waiting for stop event
        while not stop_event.is_set():
            time.sleep(0.1)

        # Cleanup
        logger.info(f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Stopping...")

        # Stop any running dance mode
        if state.current_mode and state.current_mode.running:
            logger.info(
                f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Stopping current mode: {state.current_mode.MODE_ID}"
            )
            # Run async stop in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(state.current_mode.stop())
                logger.info(
                    f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Mode stopped"
                )
            finally:
                loop.close()

        # Reset safety mixer
        if state.safety_mixer:
            logger.info(
                f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Resetting safety mixer"
            )
            state.safety_mixer.reset()

        # Signal server to shutdown
        logger.info(f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Stopping server")
        server.should_exit = True
        server_thread.join(timeout=5.0)
        logger.info(
            f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Server stopped (joined: {not server_thread.is_alive()})"
        )

        logger.info(f"[ReachyDanceDuo] {time.strftime('%H:%M:%S')} Stopped")


if __name__ == "__main__":
    instance = ReachyDanceDuo()
    instance.wrapped_run()
