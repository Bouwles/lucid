"""Record a built-in game with its autopilot into the standard raw format (docs/formats.md).

python scripts/record_web_game.py --game drift-kit --episodes 200 --workers 4 [--size 128]

It needs the app's Vite dev server (it starts one if nothing is listening on port 1420), runs
headless Chrome workers on app/record.html, and writes data/raw/<game>_<time>/ with one
episode folder per episode. Then build a dataset as usual with scripts/build_dataset.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from lucid.data.format import EpisodeMeta, SessionMeta, Step, frame_path, write_session

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
EPISODES_PER_WORKER_OFFSET = 100_000  # worker w writes episodes w*100000 + i
# Buttons each game maps (configs/games.yaml): left 1, right 2, up 4, down 8, jump 16, a 32, b 64.
MASKS = {
    "drift-kit": 1 | 2 | 4 | 8 | 32,
    "bramble": 1 | 2 | 16 | 32,
    "outpost": 1 | 2 | 4 | 8 | 16 | 32 | 64,
}


class Collector:
    def __init__(self, session_dir: Path, game: str, size: int, token: str, logic_hz: int = 30):
        self.dir = session_dir
        self.game = game
        self.size = size
        self.token = token
        self.dt = 1 / logic_hz
        self.lock = threading.Lock()
        self.open: dict[tuple[int, int], dict] = {}
        self.done_workers: set[int] = set()
        self.errors: list[str] = []
        self.episodes_written = 0
        self.ticks = 0

    def ep_id(self, worker: int, episode: int) -> str:
        return f"ep_{worker * EPISODES_PER_WORKER_OFFSET + episode:06d}"

    def episode_start(self, d: dict) -> None:
        key = (d["worker"], d["episode"])
        ep_dir = self.dir / "episodes" / self.ep_id(*key)
        (ep_dir / "frames").mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.open[key] = {
                "dir": ep_dir,
                "seed": d["seed"],
                "source": d["action_source"],
                "n": 0,
                "steps": open(ep_dir / "steps.jsonl", "w"),
                "ended": False,
            }

    def steps(self, body: bytes) -> None:
        (hlen,) = struct.unpack("<I", body[:4])
        header = json.loads(body[4 : 4 + hlen])
        ep = self.open[(header["worker"], header["episode"])]
        px = np.frombuffer(body[4 + hlen :], np.uint8).reshape(-1, self.size, self.size, 3)
        if len(px) != len(header["steps"]):
            raise ValueError("frame count does not match step count")
        for s, frame in zip(header["steps"], px, strict=True):
            if s["step"] != ep["n"]:
                raise ValueError(f"expected step {ep['n']}, got {s['step']}")
            Image.fromarray(frame).save(frame_path(ep["dir"], s["step"]), compress_level=1)
            state = {
                k: s["state"][k]
                for k in ("lane", "x", "y", "vertical", "alive", "speed", "distance")
            }
            state.update({k: v for k, v in s["state"].items() if k not in state})
            st = Step(
                s["step"],
                s["time"],
                int(s["buttons"]),
                tuple(s["look"]),
                state,
                bool(s["episode_end"]),
            )
            ep["steps"].write(json.dumps(st.to_json()) + "\n")
            ep["n"] += 1
            ep["ended"] = ep["ended"] or st.episode_end
        with self.lock:
            self.ticks += len(px)

    def episode_end(self, d: dict) -> None:
        key = (d["worker"], d["episode"])
        ep = self.open.pop(key)
        ep["steps"].close()
        meta = EpisodeMeta(
            episode_id=self.ep_id(*key),
            session_id=self.dir.name,
            game_id=self.game,
            seed=int(ep["seed"]),
            num_steps=ep["n"],
            action_source=ep["source"],
            recording_resolution=(self.size, self.size),
            logic_timestep=self.dt,
            recorder_version="0.2.0-webgame",
            game_build_version="0.2.0",
            corrupted=not ep["ended"],
            corruption_reason=None if ep["ended"] else "episode_end missing",
        )
        (ep["dir"] / "meta.json").write_text(json.dumps(meta.to_json(), indent=2))
        with self.lock:
            self.episodes_written += 1

    def make_handler(self):
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    if self.headers.get("X-Token") != collector.token:
                        raise PermissionError("bad token")
                    if self.path == "/steps":
                        collector.steps(body)
                    else:
                        d = json.loads(body)
                        {
                            "/episode_start": collector.episode_start,
                            "/episode_end": collector.episode_end,
                            "/done": lambda d: collector.done_workers.add(d["worker"]),
                            "/error": lambda d: collector.errors.append(d["error"]),
                        }[self.path](d)
                    self.send_response(200)
                    self._cors()
                    self.end_headers()
                    self.wfile.write(b"ok")
                except Exception as e:  # report to the page so it retries or stops
                    self.send_response(500)
                    self._cors()
                    self.end_headers()
                    self.wfile.write(str(e).encode())

        return Handler


def port_open(port: int, host: str = "localhost") -> bool:
    """True if something listens on the port. Vite may listen on ::1 only, so try every address."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


async def open_tab(cdp_port: int, url: str) -> None:
    # The target URL must be percent-encoded, or Chrome cuts it at the first "&".
    req = urllib.request.Request(
        f"http://127.0.0.1:{cdp_port}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT"
    )
    urllib.request.urlopen(req).read()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--game", required=True, choices=["drift-kit", "bramble", "outpost"])
    p.add_argument("--episodes", type=int, default=100, help="episodes per worker")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=128, help="recording resolution (square)")
    p.add_argument("--seed", type=int, default=1, help="first seed; worker w uses seed + w*1e6")
    p.add_argument("--max-ticks", type=int, default=4000)
    p.add_argument(
        "--random", type=float, default=0.0, help="share of time on random held inputs (0 to 1)"
    )
    p.add_argument(
        "--idle", type=float, default=0.15, help="share of random holds with no key pressed"
    )
    p.add_argument("--out", default="data/raw")
    p.add_argument(
        "--human", action="store_true", help="not supported here; human sessions come from the app"
    )
    args = p.parse_args()

    vite = None
    if not port_open(1420):
        vite = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ROOT / "app",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(120):
            if port_open(1420):
                break
            time.sleep(0.5)
        else:
            raise SystemExit("the Vite dev server didn't start on port 1420")

    session_id = f"{args.game}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    session_dir = ROOT / args.out / session_id
    write_session(
        session_dir,
        SessionMeta(
            session_id=session_id,
            game_id=args.game,
            recording_resolution=(args.size, args.size),
            logic_timestep=1 / 30,
            recorder_version="0.2.0-webgame",
            game_build_version="0.2.0",
            created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            unity_version=None,
        ),
    )
    token = secrets.token_urlsafe(16)
    collector = Collector(session_dir, args.game, args.size, token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), collector.make_handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cport = server.server_address[1]

    profile = Path(tempfile.mkdtemp(prefix="lucid-rec-"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        cdp_port = s.getsockname()[1]
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={profile}",
            "--enable-gpu",
            "--ignore-gpu-blocklist",
            "--use-angle=metal",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if port_open(cdp_port, "127.0.0.1"):
                break
            time.sleep(0.1)
        for w in range(args.workers):
            url = (
                f"http://localhost:1420/record.html?game={args.game}&collector=http://127.0.0.1:{cport}"
                f"&token={token}&worker={w}&episodes={args.episodes}&seed={args.seed + w * 1_000_000}"
                f"&size={args.size}&maxTicks={args.max_ticks}&random={args.random}&idle={args.idle}&mask={MASKS[args.game]}&look={int(args.game == 'outpost')}"
            )
            asyncio.run(open_tab(cdp_port, url))
        t0 = time.time()
        while len(collector.done_workers) < args.workers and not collector.errors:
            time.sleep(2)
            if collector.ticks == 0 and time.time() - t0 > 90:
                raise SystemExit(
                    "no frames arrived in 90 s; open the recorder page by hand to see its error"
                )
            el = time.time() - t0
            print(
                f"\r{collector.episodes_written} episodes, {collector.ticks} ticks, "
                f"{collector.ticks / max(el, 1e-6):.0f} ticks/s, {el / 60:.1f} min",
                end="",
                flush=True,
            )
        print()
        if collector.errors:
            raise SystemExit("a recorder page failed:\n" + collector.errors[0])
        print(
            f"wrote {session_dir} ({collector.episodes_written} episodes, {collector.ticks} ticks)"
        )
    finally:
        chrome.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        if vite is not None:
            vite.terminate()


if __name__ == "__main__":
    main()
