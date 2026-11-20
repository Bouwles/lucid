"""The engine's HTTP and WebSocket API (docs/engine-api.md). Localhost only, token required."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import time
from importlib import resources

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from lucid_engine import __version__
from lucid_engine.dream import Dream, Sharpener
from lucid_engine.state import EngineState
from PIL import Image

log = logging.getLogger("lucid.engine")
APP_ORIGINS = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
]
MEDIA_TYPES = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
}


def encode_webp(frame, channel: int = 0) -> bytes:
    buf = io.BytesIO()
    buf.write(bytes([channel]))
    Image.fromarray(frame).save(buf, format="WEBP", lossless=True, quality=50, method=0)
    return buf.getvalue()


def _token_ok(given: str | None, token: str) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), token.encode())


def create_app(state: EngineState, token: str) -> FastAPI:
    app = FastAPI(
        title="LUCID engine", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )
    started = time.monotonic()

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.method != "OPTIONS":
            auth = request.headers.get("authorization", "")
            given = (
                auth[7:]
                if auth.lower().startswith("bearer ")
                else request.query_params.get("token")
            )
            if not _token_ok(given, token):
                return JSONResponse({"error": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    # Added after the token check so it wraps it: preflight and 401s still carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_ORIGINS,
        # Any local dev server may call in; the per-launch token is what actually protects the API.
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.monotonic() - started, 1),
        }

    @app.get("/hardware")
    async def hardware():
        return await asyncio.to_thread(state.hardware)

    @app.get("/games")
    def games():
        return state.games()

    @app.get("/games/{game_id}")
    def game(game_id: str):
        try:
            return state.game_entry(state.catalog_game(game_id))
        except KeyError:
            raise HTTPException(404, "no such game") from None

    @app.get("/media/{name}")
    def media(name: str):
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(404)
        f = resources.files("lucid_engine").joinpath("media", name)
        if not f.is_file():
            raise HTTPException(404)
        ext = name[name.rfind(".") :].lower()
        return Response(
            f.read_bytes(),
            media_type=MEDIA_TYPES.get(ext, "application/octet-stream"),
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/library")
    def library():
        return state.library()

    @app.delete("/library/{kind}/{item_id}")
    def delete(kind: str, item_id: str):
        try:
            state.delete(kind, item_id)
        except KeyError:
            raise HTTPException(404, "not found") from None
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        return state.library()

    @app.get("/jobs")
    def jobs():
        return state.jobs.list()

    @app.post("/jobs")
    async def submit(request: Request):
        body = await request.json()
        try:
            return state.jobs.submit(body["kind"], body.get("params") or {})
        except KeyError as e:
            raise HTTPException(400, f"bad job: {e}") from None

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        try:
            return state.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404) from None

    @app.post("/jobs/{job_id}/{action}")
    def control(job_id: str, action: str):
        fn = {
            "pause": state.jobs.pause,
            "resume": state.jobs.resume,
            "cancel": state.jobs.cancel,
        }.get(action)
        if fn is None:
            raise HTTPException(404)
        try:
            return fn(job_id)
        except KeyError:
            raise HTTPException(404) from None

    @app.get("/settings")
    def settings():
        return state.settings()

    @app.put("/settings")
    async def put_settings(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "settings must be an object")
        return state.update_settings(body)

    @app.websocket("/dream")
    async def dream(ws: WebSocket):
        if not _token_ok(ws.query_params.get("token"), token):
            await ws.close(code=4401)
            return
        await ws.accept()
        installed = [p for p in state.packs.values()]
        await ws.send_text(
            json.dumps(
                {
                    "t": "hello",
                    "games": [
                        {"name": g, "id": gid}
                        for g, gid in sorted(
                            {(g, gid) for p in installed for g, gid in p.games.items()}
                        )
                    ],
                    "blend_available": any(len(p.games) > 1 for p in installed),
                    "autoplay_available": sorted(
                        {g for p in installed for g in p.games if p.has_policy_for(g)}
                    ),
                    # games whose dream comes from a multi-game model: those can switch in place and blend
                    "blend_games": sorted(
                        {
                            g
                            for p in installed
                            if len(p.games) > 1
                            for g in p.games
                            if state.pack_for(g) is p
                        }
                    ),
                    "sharp_available": state.sharp_path is not None,
                    "resolution": installed[0].meta["resolution"] if installed else [64, 64],
                    "sampler_steps": int(installed[0].meta["sampler"]["steps"]) if installed else 3,
                    "side_by_side_available": False,
                }
            )
        )
        s = {
            "buttons": 0,
            "look": (0.0, 0.0),
            "dream": None,
            "paused": False,
            "fps": 15.0,
            "reset": None,
            "steps": state.settings().get("sampler_steps"),
            "autoplay": False,
            "sharp": False,
        }

        async def receive():
            while True:
                msg = json.loads(await ws.receive_text())
                t = msg.get("t")
                if t == "act":
                    s["buttons"] = int(msg.get("buttons", 0)) & 0xFF
                    lk = msg.get("look") or [0, 0]
                    s["look"] = (float(lk[0]), float(lk[1]))
                elif t == "reset":
                    s["reset"] = msg.get("game")
                elif t == "set_game":
                    g, d = msg.get("game"), s["dream"]
                    # Same model: change game mid-dream so it morphs. Otherwise start that game's dream.
                    if d is not None and g in d.pack.games:
                        d.set_game(g)
                    else:
                        s["reset"] = g
                elif t == "config" and "sampler_steps" in msg:
                    s["steps"] = max(1, min(16, int(msg["sampler_steps"])))
                    if s["dream"] is not None:
                        s["dream"].steps = s["steps"]
                elif t == "stream":
                    s["fps"] = max(1.0, min(60.0, float(msg.get("fps", 15))))
                elif t == "quality":
                    s["sharp"] = msg.get("value") == "sharp" and state.sharp_path is not None
                    if s["dream"] is not None:
                        s["dream"].sharpener = Sharpener(state.sharp_path) if s["sharp"] else None
                elif t == "autoplay":
                    s["autoplay"] = bool(msg.get("on"))
                elif t == "pause":
                    s["paused"] = True
                elif t == "resume":
                    s["paused"] = False
                elif t == "blend":
                    d = s["dream"]
                    try:
                        if msg.get("a") is None:
                            if d is not None:
                                d.set_game(d.game)
                        elif d is None:
                            raise KeyError("start a dream first")
                        else:
                            d.set_blend(msg["a"], msg["b"], float(msg.get("alpha", 0.5)))
                    except KeyError as e:
                        await ws.send_text(json.dumps({"t": "error", "msg": str(e).strip("'")}))
                elif t == "side_by_side":
                    await ws.send_text(
                        json.dumps({"t": "error", "msg": "side by side isn't available yet"})
                    )

        async def generate():
            last_stats = time.monotonic()
            while True:
                t0 = time.monotonic()
                if s["reset"] is not None:
                    gid, s["reset"] = s["reset"], None
                    pack = state.pack_for(gid)
                    if pack is None:
                        await ws.send_text(
                            json.dumps({"t": "error", "msg": f"no dream installed for {gid}"})
                        )
                    else:
                        s["dream"] = Dream(pack, steps=s["steps"], game=gid)
                        if s["sharp"]:
                            s["dream"].sharpener = Sharpener(state.sharp_path)
                        frame = await asyncio.to_thread(s["dream"].reset)
                        await ws.send_bytes(encode_webp(frame))
                d = s["dream"]
                if d is not None and not s["paused"]:
                    if s["autoplay"] and d.pack.has_policy_for(d.game):
                        # Dream mode: a policy picks the keys from the dream's own frames.
                        b, lk = await asyncio.to_thread(d.policy_action)
                    else:
                        b, lk = s["buttons"], s["look"]
                    frame = await asyncio.to_thread(d.step, b, lk)
                    await ws.send_bytes(encode_webp(frame))
                    if time.monotonic() - last_stats > 1.0:
                        last_stats = time.monotonic()
                        await ws.send_text(
                            json.dumps(
                                {
                                    "t": "stats",
                                    "fps": round(min(d.fps, s["fps"]), 1),
                                    "frame_ms": round(d.frame_ms, 1),
                                    "sampler_steps": d.steps,
                                }
                            )
                        )
                await asyncio.sleep(max(0.0, 1.0 / s["fps"] - (time.monotonic() - t0)))

        tasks = [asyncio.create_task(receive()), asyncio.create_task(generate())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect | RuntimeError):
                    log.warning("dream session ended with %r", exc)
        finally:
            log.info("dream session closed")

    return app
