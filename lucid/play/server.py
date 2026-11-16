"""FastAPI server for the browser demo. Protocol: docs/formats.md.

One DreamEngine per connected player (sharing one model's weights), a cap on players, idle
timeouts, and a health check. Nothing personal about visitors is logged: only counts and timings.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from lucid.play.engine import DreamEngine

log = logging.getLogger("lucid.server")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def encode_webp(frame, channel: int = 0) -> bytes:
    """uint8 [H, W, 3] -> channel byte + lossless WebP bytes."""
    buf = io.BytesIO()
    buf.write(bytes([channel]))
    Image.fromarray(frame).save(buf, format="WEBP", lossless=True, quality=50, method=0)
    return buf.getvalue()


class _Stats:
    def __init__(self):
        self.players = 0
        self.total_connections = 0
        self.frame_ms: list[float] = []

    def add_frame(self, ms: float) -> None:
        self.frame_ms.append(ms)
        self.frame_ms = self.frame_ms[-500:]

    @property
    def avg_frame_ms(self) -> float:
        return sum(self.frame_ms) / len(self.frame_ms) if self.frame_ms else 0.0


def create_app(
    engine_factory: Callable[[], DreamEngine],
    max_players: int = 4,
    idle_timeout: float = 120.0,
    target_fps: float = 20.0,
    web_dir: Path | None = WEB_DIR,
) -> FastAPI:
    app = FastAPI(title="LUCID")
    stats = _Stats()
    gpu_lock = asyncio.Lock()  # one generation at a time keeps per-frame latency predictable

    @app.get("/health")
    def health():
        return JSONResponse(
            {
                "status": "ok",
                "players": stats.players,
                "max_players": max_players,
                "avg_frame_ms": round(stats.avg_frame_ms, 2),
                "total_connections": stats.total_connections,
            }
        )

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        if stats.players >= max_players:
            await socket.send_text(json.dumps({"t": "busy", "max_players": max_players}))
            await socket.close()
            return
        stats.players += 1
        stats.total_connections += 1
        log.info("player connected (%d/%d)", stats.players, max_players)
        engine = await asyncio.to_thread(engine_factory)
        state = {
            "buttons": 0,
            "look": (0.0, 0.0),
            "playing": False,
            "side_by_side": False,
            "last_msg": time.monotonic(),
            "reset": None,
        }
        await socket.send_text(
            json.dumps(
                {
                    "t": "hello",
                    "games": [{"name": g, "id": i} for g, i in engine.games.items()],
                    "resolution": list(
                        engine.resolution
                        if engine.upsampler is None
                        else engine.ups_cfg.data.resolution
                    ),
                    "sampler_steps": engine.sampler_steps,
                    "side_by_side_available": engine._load_starts().get(engine.game)
                    not in (None, []),
                }
            )
        )

        async def receive():
            while True:
                msg = json.loads(await socket.receive_text())
                state["last_msg"] = time.monotonic()
                t = msg.get("t")
                try:
                    if t == "act":
                        state["buttons"] = int(msg.get("buttons", 0)) & 0xFF
                        lk = msg.get("look") or [0, 0]
                        state["look"] = (float(lk[0]), float(lk[1]))
                    elif t == "reset":
                        state["reset"] = msg.get("game") or engine.game
                    elif t == "set_game":
                        engine.set_game(msg["game"])
                    elif t == "blend":
                        if msg.get("a") is None:
                            engine.clear_blend()
                        else:
                            engine.set_blend(msg["a"], msg["b"], float(msg.get("alpha", 0.5)))
                    elif t == "config":
                        engine.sampler_steps = int(msg["sampler_steps"])
                    elif t == "side_by_side":
                        state["side_by_side"] = bool(msg.get("on"))
                        state["reset"] = engine.game
                except (KeyError, ValueError, TypeError) as e:
                    await socket.send_text(json.dumps({"t": "error", "msg": str(e)}))

        async def generate():
            last_stats = time.monotonic()
            period = 1.0 / target_fps
            while True:
                t0 = time.monotonic()
                if t0 - state["last_msg"] > idle_timeout:
                    await socket.close(code=1000, reason="idle")
                    return
                if state["reset"] is not None:
                    game, state["reset"] = state["reset"], None
                    async with gpu_lock:
                        frame = await asyncio.to_thread(engine.reset, game)
                    state["playing"] = True
                    await socket.send_bytes(encode_webp(frame))
                if state["playing"]:
                    async with gpu_lock:
                        if state["side_by_side"]:
                            pair = await asyncio.to_thread(engine.step_replay)
                            if pair is None:  # the recorded episode ended: start another
                                state["reset"] = engine.game
                                continue
                            real, frame = pair
                        else:
                            real = None
                            frame = await asyncio.to_thread(
                                engine.step, {"buttons": state["buttons"], "look": state["look"]}
                            )
                    stats.add_frame(engine.frame_ms)
                    if real is not None:
                        await socket.send_bytes(encode_webp(real, channel=1))
                    await socket.send_bytes(encode_webp(frame))
                if time.monotonic() - last_stats > 1.0:
                    last_stats = time.monotonic()
                    await socket.send_text(
                        json.dumps(
                            {
                                "t": "stats",
                                "fps": round(engine.fps_estimate, 1),
                                "frame_ms": round(engine.frame_ms, 1),
                                "sampler_steps": engine.sampler_steps,
                            }
                        )
                    )
                await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

        tasks = [asyncio.create_task(receive()), asyncio.create_task(generate())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect | RuntimeError):
                    log.warning("session ended with %s", type(exc).__name__)
        finally:
            stats.players -= 1
            log.info(
                "player left (%d/%d), avg frame %.1f ms",
                stats.players,
                max_players,
                stats.avg_frame_ms,
            )

    if web_dir is not None and web_dir.exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
