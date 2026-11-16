# The LUCID desktop app

The app is two programs. The shell in `app/` (Tauri 2, Svelte 5, TypeScript) draws the interface. The engine in `engine/` (Python) runs dreams, jobs and hardware checks. They talk over `127.0.0.1` only, as described in [engine-api.md](engine-api.md).

## Layout

```
app/
  src/                 interface (Svelte)
    lib/engine.ts      HTTP + WebSocket client for the engine
    lib/shell.ts       Tauri commands, with browser fallbacks
    screens/           first launch, home, game page, dream player, teach, library, settings
  src-tauri/           Rust shell
    src/lib.rs         starts and stops the engine, launches games, smoke test
    binaries/          lucid-engine-<target triple> (the packaged engine, not committed)
    resources/models/  dream packs bundled with the app (test-track)
  scripts/make_icon.py draws the app icon
```

## Running it during development

You need Node 24, Rust (`~/.cargo/bin`), and the repo's `.venv` with the engine's dependencies.

```bash
cd app
npm install
export PATH="$HOME/.cargo/bin:$PATH"
npx tauri dev
```

In debug builds the shell runs the engine from source: `<repo>/.venv/bin/python -m lucid_engine` with `PYTHONPATH=<repo>/engine:<repo>`. It finds the repo from the source tree; set `LUCID_REPO` to point somewhere else. Set `LUCID_ENGINE_BIN=<path>` to try the packaged engine binary in a debug build instead.

### Interface only, in a browser

Useful for working on screens without rebuilding Rust:

```bash
# terminal 1: start the engine and copy the port and token from its ready line
cd <repo>
PYTHONPATH=engine:. .venv/bin/python -m lucid_engine --data-dir /tmp/lucid-dev --bundled-models app/src-tauri/resources/models

# terminal 2
cd app
VITE_ENGINE_PORT=<port> VITE_ENGINE_TOKEN=<token> npm run dev
# open http://localhost:1420
```

Finder buttons and launching games only work in the desktop app.

## Building the app

Two pieces aren't in git because they're large, so make them first:

```bash
# the packaged engine (a clean build environment, no PyTorch)
.venv/bin/python engine/build_sidecar.py

# the bundled test-track dream pack, from a trained checkpoint
.venv/bin/python scripts/export_dream_pack.py --checkpoint runs/<run>/best.pt \
    --id test-track --game synthetic --out app/src-tauri/resources/models
```

Then:

```bash
cd app
export PATH="$HOME/.cargo/bin:$PATH"
npx tauri build
```

This needs the packaged engine at `app/src-tauri/binaries/lucid-engine-<target triple>` (for example `lucid-engine-aarch64-apple-darwin`). If it's missing, `build.rs` writes a placeholder so the Rust code still compiles, but the built app won't start its engine.

Output on macOS:

- `app/src-tauri/target/release/bundle/macos/LUCID.app`
- `app/src-tauri/target/release/bundle/dmg/LUCID_0.2.0_aarch64.dmg`

The builds aren't signed. The first time you open the app on macOS, right-click it and choose Open.

## Where it keeps things

`<platform data folder>/LUCID/`, which on macOS is `~/Library/Application Support/LUCID/`. Models, games, projects, jobs, `settings.json` and `logs/engine.log` live there.

## Smoke test

```bash
LUCID_SMOKE_TEST=1 app/src-tauri/target/release/bundle/macos/LUCID.app/Contents/MacOS/lucid
echo $?
```

The app opens the first installed dream, waits for 30 drawn frames (30 s limit), writes `logs/smoke.json` in the data folder and exits with 0 on success or 1 on failure. If the engine doesn't start, that's reported as a failure too. A backstop in the shell exits with 1 after 150 s if the page never reports.

## How the shell looks after the engine

- It reads the engine's stdout until `LUCID_ENGINE_READY {...}`, waiting up to 60 s.
- It keeps the engine's stdin open. On quit, or when the window closes, it closes stdin, waits up to 3 s, then sends SIGTERM and finally kills it. If the app itself crashes, the pipe closes anyway and the engine exits on its own.
- If the engine dies while the app is running, the interface shows an error screen with the last lines of its output, the log path, and a Retry button that restarts the engine.

## Icon

`app/scripts/make_icon.py` draws the mark: a circle that's smooth on the left and pixelated on the right, the real frame and its dream. To regenerate it:

```bash
cd app
../.venv/bin/python scripts/make_icon.py
npx tauri icon src-tauri/icons/source.png
rm -rf src-tauri/icons/android src-tauri/icons/ios
```
