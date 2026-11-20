"""Test pixel policies in the REAL game: random holds, behaviour cloning, and dream-trained.

python scripts/eval_policy_real.py --game bramble --policy bc=<bc policy.pt> --policy dream=<dream policy.pt> --episodes 20

Each policy plays the same seeds of the real game in headless Chrome (app/remote.html), seeing only
64x64 frames at the dream's rate. Reports distance reached and how many episodes ended alive.
"""

import argparse
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_web_game import CHROME, MASKS, open_tab, port_open  # noqa: E402

from lucid.data.dataset import to_model_space  # noqa: E402
from lucid.policy import Policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Player:
    def __init__(self, kind: str, path: str | None, mask: int, k: int = 4):
        self.kind, self.mask, self.k = kind, mask, k
        self.rng = np.random.default_rng(0)
        self.frames: list[np.ndarray] = []
        self.held, self.left = 0, 0
        self.pol = None
        if path:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.pol = Policy(ck["k"])
            self.pol.load_state_dict(ck["state"])
            self.pol.eval()
            self.k = ck["k"]

    def reset(self):
        self.frames, self.left = [], 0

    def act(self, frame: np.ndarray) -> int:
        self.frames = (self.frames + [frame])[-self.k :]
        if self.pol is None:  # random held buttons, the same kind used when recording
            if self.left <= 0:
                bits = [1 << i for i in range(8) if self.mask & (1 << i)]
                self.held = (
                    int(
                        np.bitwise_or.reduce(
                            self.rng.choice(bits, size=int(self.rng.integers(0, 3)))
                        )
                    )
                    if bits
                    else 0
                )
                self.left = int(self.rng.integers(2, 12))
            self.left -= 1
            return self.held & self.mask
        while len(self.frames) < self.k:
            self.frames.insert(0, self.frames[0])
        x = to_model_space(torch.from_numpy(np.stack(self.frames)[None]).permute(0, 1, 4, 2, 3))
        with torch.no_grad():
            probs = torch.sigmoid(self.pol(x)[0])[0].numpy()
        return int(sum(1 << i for i, p in enumerate(probs) if p > 0.5)) & self.mask


def run(game: str, player: Player, episodes: int, seed: int, size: int = 64) -> list[dict]:
    token = secrets.token_urlsafe(12)
    results, done, errors = [], threading.Event(), []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            out = b"{}"
            if self.headers.get("X-Token") == token:
                if self.path == "/act":
                    b = player.act(np.frombuffer(body, np.uint8).reshape(size, size, 3))
                    out = json.dumps({"buttons": b, "look": [0, 0]}).encode()
                elif self.path == "/start":
                    player.reset()
                elif self.path == "/end":
                    results.append(json.loads(body))
                elif self.path == "/done":
                    done.set()
                elif self.path == "/error":
                    errors.append(json.loads(body)["error"])
                    done.set()
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    profile = Path(tempfile.mkdtemp(prefix="lucid-eval-"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        cdp = s.getsockname()[1]
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={cdp}",
            f"--user-data-dir={profile}",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if port_open(cdp, "127.0.0.1"):
                break
            time.sleep(0.1)
        import asyncio

        url = (
            f"http://localhost:1420/remote.html?game={game}&server=http://127.0.0.1:{srv.server_address[1]}"
            f"&token={token}&episodes={episodes}&seed={seed}&size={size}"
        )
        asyncio.run(open_tab(cdp, url))
        done.wait(timeout=3600)
        if errors:
            raise SystemExit("the game page failed:\n" + errors[0])
    finally:
        chrome.terminate()
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--game", required=True, choices=list(MASKS))
    p.add_argument(
        "--policy", action="append", default=[], help="name=path/to/policy.pt (repeatable)"
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=9000, help="first seed; held-out from training")
    p.add_argument("--out", default=None)
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
    players = {"random": None, **dict(s.split("=", 1) for s in args.policy)}
    summary = {}
    try:
        for name, path in players.items():
            res = run(args.game, Player(name, path, MASKS[args.game]), args.episodes, args.seed)
            dist = np.array([r["distance"] for r in res])
            summary[name] = {
                "episodes": len(res),
                "mean_distance": float(dist.mean()),
                "median_distance": float(np.median(dist)),
                "alive_at_end": float(np.mean([r["alive"] for r in res])),
                "per_episode": res,
            }
            print(
                f"{name:>8}: mean distance {dist.mean():.1f} (median {np.median(dist):.1f}) over {len(res)} episodes"
            )
    finally:
        if vite is not None:
            vite.terminate()
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
