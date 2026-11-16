"""Start the engine: lucid-engine --data-dir <dir> [--bundled-models <dir>] [--port 0].

Prints one line, `LUCID_ENGINE_READY {"port": ..., "token": ..., "version": ...}`, once it is
serving on 127.0.0.1. Exits on SIGTERM, Ctrl+C, or when stdin closes (so it never outlives the app).
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import signal
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="lucid-engine", description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--bundled-models", default=None)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--token", default=None, help="for development only; normally random per launch")
    args = p.parse_args(argv)

    import uvicorn
    from lucid_engine import __version__
    from lucid_engine.app import create_app
    from lucid_engine.state import EngineState

    data = Path(args.data_dir).expanduser()
    (data / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [
        RotatingFileHandler(data / "logs" / "engine.log", maxBytes=2_000_000, backupCount=2),
        logging.StreamHandler(sys.stderr),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger("lucid.engine")

    state = EngineState(data, Path(args.bundled_models) if args.bundled_models else None)
    token = args.token or secrets.token_urlsafe(32)
    app = create_app(state, token)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))  # never an external interface
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="warning", lifespan="off", ws="websockets")
    )
    server.install_signal_handlers = lambda: (
        None
    )  # handled below; the server runs off the main thread

    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            sys.exit("engine failed to start")
        time.sleep(0.02)

    def stop(*_):
        server.should_exit = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if not sys.stdin.isatty():

        def watch_stdin():
            try:
                while sys.stdin.read(1024):
                    pass
            except (OSError, ValueError):
                pass
            stop()

        threading.Thread(target=watch_stdin, daemon=True).start()

    log.info(
        "engine %s serving on 127.0.0.1:%d, %d dream pack(s)", __version__, port, len(state.packs)
    )
    print(
        "LUCID_ENGINE_READY " + json.dumps({"port": port, "token": token, "version": __version__}),
        flush=True,
    )
    while thread.is_alive():
        thread.join(0.2)
    state.jobs.shutdown()
    log.info("engine stopped")


if __name__ == "__main__":
    main()
