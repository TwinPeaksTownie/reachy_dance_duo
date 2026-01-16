"""Main entry point for the Reachy Dance Suite.

Provides both a web UI (default) and a CLI for starting dance modes.

Usage:
    # UI Mode
    python -m reachy_dance_suite

    # CLI Mode - starts dancing immediately
    python -m reachy_dance_suite --mode live_groove
    python -m reachy_dance_suite --mode synthwave_serenade
    python -m reachy_dance_suite --mode beat_bandit --url "https://youtube.com/..."
"""

import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reachy's Ultra Dance Mix 9000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m reachy_dance_suite                                    # Start in UI mode (idle)
    python -m reachy_dance_suite --mode live_groove                 # Start Live Groove
    python -m reachy_dance_suite --mode beat_bandit --url "https://youtube.com/watch?v=..."
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["live_groove", "synthwave_serenade", "beat_bandit"],
        help="Start in this mode immediately (skip idle state)",
    )
    parser.add_argument(
        "--url",
        help="YouTube URL for beat_bandit mode",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for web server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web server (default: 8000)",
    )

    return parser.parse_args()


def run_cli(mode: str, url: Optional[str] = None):
    """Run in CLI mode - start dancing immediately without web UI."""
    from reachy_mini import ReachyMini
    from .behaviors.base import DanceMode
    from .behaviors.connected_choreographer import ConnectedChoreographer
    from .behaviors.live_groove import LiveGroove
    from .behaviors.synthwave_serenade import SynthwaveSerenade
    from .config import get_default_safety_config
    from .core.safety_mixer import SafetyMixer

    logger.info(f"Starting Reachy Dance Suite in CLI mode ({mode})")
    logger.info("Press Ctrl+C to stop")

    try:
        mini = ReachyMini()
        logger.info("ReachyMini connected")

        safety_config = get_default_safety_config()
        safety_mixer = SafetyMixer(safety_config, mini)
        logger.info("SafetyMixer initialized")

        # Initialize MotionController for synthwave_serenade mode
        motion_controller = None
        if mode == "synthwave_serenade":
            from .core.motion_controller import MotionController

            motion_controller = MotionController(mini, enabled=True)
            if motion_controller.init_head_tracker(device="mps"):
                logger.info("MotionController head tracker initialized (MPS)")
            else:
                logger.error(
                    "MotionController head tracker failed - face tracking disabled"
                )

        # Create and start the requested mode
        dance_mode: DanceMode

        if mode == "synthwave_serenade":
            if motion_controller is None:
                logger.error("MotionController failed to initialize")
                return
            dance_mode = SynthwaveSerenade(safety_mixer, mini, motion_controller)
        elif mode == "live_groove":
            dance_mode = LiveGroove(safety_mixer, mini)
        elif mode == "beat_bandit":
            bb_mode = ConnectedChoreographer(
                safety_mixer, mini, None
            )  # ytmusic not needed for direct URL
            dance_mode = bb_mode
            if url:
                import asyncio

                loop = asyncio.get_event_loop()
                success = loop.run_until_complete(bb_mode.set_youtube_url(url))
                if not success:
                    logger.error("Failed to initialize YouTube track")
                    return
        else:
            logger.error(f"Unknown mode: {mode}")
            return

        # Start dancing
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(dance_mode.start())

    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.exception(f"Error in CLI mode: {e}")
    finally:
        if "dance_mode" in locals():
            import asyncio

            loop = asyncio.get_event_loop()
            loop.run_until_complete(dance_mode.stop())


def main():
    """Run the application."""
    args = parse_args()

    if args.mode:
        run_cli(args.mode, args.url)
    else:
        # Import app here to avoid early initialization
        from .app import app
        import uvicorn

        logger.info(f"Starting web UI on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
