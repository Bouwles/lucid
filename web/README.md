# web/

The browser demo. Plain HTML, CSS and one JS module, with no build step and no outside requests, so it works offline when the FastAPI server serves it.

## Run it

```bash
python scripts/serve.py --checkpoint runs/<run>/best.pt
```

For a quick check of the UI without a trained model:

```bash
python scripts/serve.py --random-weights
```

Then open http://localhost:8000 and press Start. With random weights the frames are noise, but every control, the stats and the reconnect flow can be tested.

## Files

- `index.html`: layout, start screen, overlays (connecting, busy, disconnected), controls and touch buttons.
- `style.css`: dark theme with one accent colour.
- `main.js`: WebSocket client, input handling and frame drawing.

## How it talks to the server

The protocol is in `docs/formats.md` ("WebSocket protocol"). In short:

- The socket URL is built from the page's own address (`ws://` on http, `wss://` on https), so the same files work locally and behind a deployed host.
- After `hello`, the client sends `reset` for the chosen game, then an `act` message 20 times a second and immediately whenever the held buttons change. Sending pauses while the tab is hidden, after one last `act` with no buttons held.
- Binary messages are one channel byte (0 dream, 1 real recording) followed by a WebP image. Decoding is asynchronous, so only the newest frame per channel is drawn and older ones are dropped.
- The canvas is drawn at the model's resolution and scaled up by the largest whole number that fits, with `image-rendering: pixelated`, so pixels stay sharp.

## Controls

| Input | Button |
|---|---|
| Arrows or WASD | left, right, up, down |
| Space | jump |
| Z | action_a |
| X | action_b |

Left and right together count as neither, and the same for up and down. Phones get on-screen buttons for the d-pad and jump. The demo sends `look: [0, 0]`; mouse look is not wired up here yet.

The side-by-side checkbox only appears when the server says a real recording is available. The blend slider and the "switch mid-dream" option sit under an experiment heading because the model was never trained on blended or switched games.
