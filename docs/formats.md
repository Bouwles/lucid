# File formats and protocols

The contracts between LUCID's parts. The Unity recorder, the web recorder, `lucid/data/format.py`, `lucid/play/server.py` and `web/main.js` all follow this file. Change it only together with every side that uses it, and add a line to `docs/decisions.md`.

## Unified action

- `buttons`: an integer 0-255. Bit `i` is set when button `i` is held:

  | bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | button | left | right | up | down | jump | action_a | action_b | reserved |

- `look`: two floats in [-1, 1], the horizontal and vertical look delta for that step. `[0, 0]` for 2D games. Raw mouse deltas are clipped and divided by the per-game `look_scale` before saving.
- Left and right pressed together cancel out to neither. Same for up and down.

## Raw recordings (format_version 1)

```
data/raw/<session_id>/
  session.json
  episodes/
    ep_000000/
      meta.json
      steps.jsonl
      frames/000000.png, 000001.png, ...
```

`session_id` is `<game_id>_<yyyyMMdd_HHmmss>` in UTC. Episode folders are `ep_` plus six zero-padded digits; frame files are the six zero-padded step index. Frames are 8-bit RGB PNG (alpha is ignored if present), row 0 at the top, at the recording resolution. Lossless only: never JPEG or video.

### session.json

```json
{"format_version": 1, "session_id": "runner_20261116_101500", "game_id": "runner",
 "created_utc": "2026-11-16T10:15:00Z", "recorder_version": "0.1.0",
 "game_build_version": "1.0", "unity_version": "2022.3.40f1",
 "recording_resolution": [256, 256], "logic_timestep": 0.0166667, "frame_format": "png"}
```

`unity_version` is `null` for the web recorder.

### meta.json (written when the episode ends)

```json
{"format_version": 1, "episode_id": "ep_000000", "session_id": "runner_20261116_101500",
 "game_id": "runner", "seed": 12345, "game_build_version": "1.0", "recorder_version": "0.1.0",
 "action_source": {"name": "autopilot", "settings": {"p_random": 0.25}},
 "recording_resolution": [256, 256], "logic_timestep": 0.0166667, "look_scale": null,
 "num_steps": 812, "corrupted": false, "corruption_reason": null}
```

`action_source.name` is `human`, `autopilot`, `replay` or a more specific autopilot name. Human episodes are recognised by `name == "human"`.

### steps.jsonl (one JSON object per logic step, in order)

```json
{"step": 0, "time": 0.0, "buttons": 0, "look": [0.0, 0.0],
 "state": {"lane": 1, "x": 0.0, "y": 0.0, "vertical": "grounded", "alive": true, "speed": 8.0, "distance": 0.0},
 "episode_end": false}
```

- `step` runs 0..N-1 with no gaps, and N equals the number of frames.
- `state.vertical` is one of `grounded`, `jumping`, `sliding`, `falling`, `none`. `lane` is `null` for games without lanes.
- **Timing:** frame N is the image the player saw when action N was chosen. Action N is applied after frame N is captured, and its effect first appears in frame N+1.
- Exactly one line has `episode_end: true`, and it is the last line. On death the game shows its game-over frame for a fixed time (about 1 second) with `alive: false` and `buttons: 0`, then the episode ends.
- `state` is a label for evaluation probes and statistics only. It is never a model input.

## Processed dataset

```
data/processed/<version>/
  manifest.json        build config, git commit, splits, per-game ids and padding, measured sigma_data
  validation.md/.json  which raw episodes failed and why
  stats.md, mean_frame.png, examples.png
  <split>/<game_id>/<session_id>__<episode_id>.npz
  <split>/<game_id>/<session_id>__<episode_id>.frames.npy
  <split>/<game_id>/<session_id>__<episode_id>.full.npy   (only with full_resolution set)
```

- `.frames.npy`: `uint8 [T, H, W, 3]` at the training resolution, letterboxed (padded, never stretched) if the aspect ratio differs. Stored as plain `.npy` so it can be memory-mapped.
- `.full.npy`: the same frames at `full_resolution`, the upsampler's targets.
- `.npz`: `buttons uint8 [T, 8]` (unpacked bits), `look float16 [T, 2]`, `episode_end bool [T]`, and labels `lane int16 [T]` (-1 when there are no lanes), `vertical int8 [T]` (index into the list above), `alive bool [T]`, `speed`, `distance`, `x`, `y` (`float32 [T]`).
- Index `t` after frame skipping is logic step `t * frame_skip`. Its action summarises logic steps `t*s .. t*s+s-1` using the build's `action_rule`, and look is their mean.
- Splits are `train`, `val`, `test` and `test_human`, assigned per episode by a hash of `session/episode`, so adding sessions never moves old episodes.
- A version is immutable: the build refuses to overwrite an existing `manifest.json`.

## WebSocket protocol (browser demo)

The FastAPI server serves `web/` at `/`, the socket at `/ws`, and a health check at `/health`.

Client to server, JSON text:

- `{"t": "act", "buttons": int, "look": [x, y]}` sent at a steady 20 Hz while playing and immediately when the keys change.
- `{"t": "reset", "game": "runner"}` starts or restarts a dream of that game.
- `{"t": "set_game", "game": "platformer"}` switches game in the middle of a rollout, without a reset.
- `{"t": "blend", "a": "runner", "b": "platformer", "alpha": 0.5}` blends two game embeddings (alpha 0 means all `a`). `{"t": "blend", "a": null}` clears it.
- `{"t": "config", "sampler_steps": int}`.
- `{"t": "side_by_side", "on": bool}`.

Server to client:

- `{"t": "hello", "games": [{"name": "runner", "id": 0}], "resolution": [W, H], "sampler_steps": int, "side_by_side_available": bool}` is the first message after connecting.
- Binary messages: byte 0 is the channel (0 for a dream frame, 1 for a real frame in side-by-side mode) and the remaining bytes are a WebP image.
- `{"t": "stats", "fps": float, "frame_ms": float, "sampler_steps": int}` about once a second.
- `{"t": "busy", "max_players": int}`, after which the server closes the socket.
- `{"t": "error", "msg": str}`.
- Idle sessions are closed after a timeout.
