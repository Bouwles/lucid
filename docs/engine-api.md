# Engine API

The LUCID desktop app is two programs. The shell (Tauri, `app/`) draws the interface. The engine (Python, `engine/`) runs dreams, jobs and hardware checks. They talk over HTTP and a WebSocket on `127.0.0.1` only. This file is the contract between them.

## Launch

The shell starts the engine as a sidecar:

```
lucid-engine --data-dir <app data>/LUCID --bundled-models <app resources>/models [--port 0]
```

- It binds `127.0.0.1` only, on a random free port unless `--port` is given.
- It makes a fresh random token every launch.
- When it is ready to serve, it prints exactly one line to stdout and nothing else after it:

  ```
  LUCID_ENGINE_READY {"port": 51234, "token": "...", "version": "0.2.0"}
  ```

- Logs go to `<data dir>/logs/engine.log` (and stderr).
- It exits cleanly on SIGTERM, or when its stdin closes. The shell keeps a pipe open and closes it on quit, so the engine never outlives the app.

## Auth

Every request needs the token, either as a header `Authorization: Bearer <token>`, or as a `token` query parameter (for WebSockets, `<img>` and `<video>` tags). Anything else gets `401`. CORS allows only the Tauri app origins and the Vite dev server.

## Data folder

```
<data dir>/
  models/<id>/      downloaded or trained dream packs
  games/<id>/       downloaded game builds
  projects/<id>/    Teach LUCID projects
  jobs/<id>.json    job state (survives restarts)
  logs/engine.log
  settings.json
```

## Dream packs

A dream model ships as a folder:

```
<id>/
  dream.json      metadata (below)
  denoiser.onnx   the EDM denoiser D(x; sigma, cond), exported from PyTorch
  starts.npz      real starting contexts: frames uint8 [N, k, H, W, 3], buttons uint8 [N, k, 8], look float32 [N, k, 2]
```

`dream.json`:

```json
{"format": 1, "id": "test-track", "game": "test-track", "game_id": 0, "num_games": 1,
 "resolution": [64, 64], "context_frames": 4, "action_history": true,
 "sigma_data": 0.5, "sampler": {"steps": 3, "sigma_min": 0.002, "sigma_max": 80, "rho": 7, "context_noise": 0.1},
 "quality": "fast", "trained_from": "<run name>", "dataset_version": "...", "created_utc": "..."}
```

Bundled packs (in the app resources) are read-only. Downloaded and trained packs live in the data folder.

## HTTP endpoints

All responses are JSON unless noted.

| Method and path | What |
|---|---|
| `GET /health` | `{"status": "ok", "version": str, "uptime_s": float}` |
| `GET /hardware` | What this machine can do (below) |
| `GET /games` | The game catalogue, with what is installed (below) |
| `GET /games/{id}` | One catalogue entry |
| `GET /media/{file}` | Cover images and clips for the catalogue (binary) |
| `GET /library` | Installed models, games and projects with sizes, plus free disk space |
| `DELETE /library/{kind}/{id}` | Delete a downloaded model, game or project (`kind` is `models`, `games` or `projects`). Bundled models can't be deleted (`409`). |
| `GET /jobs` | All jobs, newest first |
| `POST /jobs` | Start a job: `{"kind": "download_model" or "benchmark", "params": {...}}` |
| `GET /jobs/{id}` | One job |
| `POST /jobs/{id}/pause`, `/resume`, `/cancel` | Control a job |
| `GET /settings` | Settings object |
| `PUT /settings` | Merge keys into settings and return the result |

### `/hardware`

```json
{"os": "macOS 26.0", "arch": "arm64", "device": "mps", "gpu": "Apple M5", "memory_gb": 32,
 "onnx_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
 "dream_fps": 85.0, "dream_fps_measured": true,
 "can_train": true, "training_pack_installed": false,
 "summary": "This Mac can play dreams at about 85 FPS. Teaching LUCID a new game needs the training pack (about 2 GB)."}
```

`dream_fps` is measured by running the bundled test model the first time it's asked for, then cached.

### `/games` entries

```json
{"id": "drift-kit", "name": "Drift Kit", "genre": "Low-poly racer",
 "tagline": "...", "description": "...", "accent": "#ff6b35",
 "status": "available" or "coming_soon",
 "controls": [{"keys": "Arrows / WASD", "action": "Steer, accelerate, brake"}],
 "game": {"downloadable": false, "installed": false, "size_mb": null},
 "dream": {"installed": true, "bundled": true, "size_mb": 6.1, "quality": ["fast"], "downloadable": false},
 "media": {"cover": "/media/test-track.png" or null, "clip": "/media/test-track.gif" or null}}
```

### Jobs

```json
{"id": "j_...", "kind": "download_model", "params": {"game": "drift-kit"},
 "state": "queued" | "running" | "paused" | "done" | "failed" | "cancelled",
 "progress": 0.42, "message": "Downloading dream model (12 of 31 MB)",
 "eta_s": 18.0, "created_utc": "...", "updated_utc": "...", "error": null}
```

Job state is written to `jobs/<id>.json` on every change. After a restart, jobs that were `running` come back as `paused`, and resuming continues where they left off (downloads resume with HTTP Range requests). Downloads only come from URLs in the catalogue (the LUCID GitHub Releases) and are checked against the catalogue's SHA-256 before installing.

## Dream WebSocket

`ws://127.0.0.1:<port>/dream?token=<token>` speaks the same protocol as the v1 browser demo (see [formats.md](formats.md#websocket-protocol-browser-demo)), with these differences:

- `reset`, `set_game` and `blend` use catalogue ids (`"test-track"`, `"drift-kit"`).
- `hello` lists only games whose dream is installed.
- The server sends a frame each time it receives an `act` (or at most `target_fps` frames a second if the client sends `{"t": "stream", "fps": 15}`). The default is streaming at 15 FPS.
- `{"t": "pause"}` and `{"t": "resume"}` stop and restart frame generation (for example while the window is hidden).
