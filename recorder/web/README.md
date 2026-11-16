# recorder/web/

A small recorder for browser games (canvas-based, like RAJIS: Uprising). It saves frames, actions and game state in the same raw format as the Unity recorder, so `scripts/validate_raw.py` and `scripts/build_dataset.py` treat both the same. The format is in `docs/formats.md`.

- `src/lucidRecorder.ts`: the source, with types.
- `dist/lucidRecorder.js`: the same code as a plain ES module, usable with no build step.

`npm run typecheck` checks the TypeScript (needs `npm install` first). The JS file is kept in sync by hand.

## Usage

The recorder does not bundle a zip library. Pass in a JSZip constructor (https://stuk.github.io/jszip/, MIT), for example from a local copy of `jszip.min.js` loaded with a script tag, which defines `window.JSZip`.

```js
import {
  createRecorder,
  KeyboardActionSource,
  installSeededMathRandom,
} from "./lucidRecorder.js";

const actions = new KeyboardActionSource(); // 2D game: no look
const recorder = createRecorder({
  canvas: document.querySelector("canvas"),
  gameId: "rajis",
  recordingSize: [256, 256],
  logicTimestep: 1 / 60,
  gameBuildVersion: "1.0",
  getState: () => ({
    lane: null, x: player.x, y: player.y,
    vertical: player.onGround ? "grounded" : "jumping",
    alive: player.alive, speed: player.speed, distance: player.distance,
  }),
  actionSource: actions,
  zip: window.JSZip,
});

const seed = 12345;
let uninstall = installSeededMathRandom(seed);
game.restart(seed);
recorder.startEpisode(seed);

function logicStep() {
  // The canvas must already show frame N here.
  const action = recorder.tick(); // captures frame N, picks action N
  game.update(action, 1 / 60);    // applies action N
  game.render();                  // draws frame N+1
  if (game.overFor >= 1.0) {      // about 1 s of game-over frames, then end
    recorder.endEpisode();
    uninstall();
  }
}
```

Then save and download:

```js
const blob = await recorder.save({ release: true });
const a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = `${recorder.sessionId}.zip`;
a.click();
```

Unzip into `data/raw/<session_id>/`. The zip has `session.json` at its root and `episodes/ep_000000/{meta.json, steps.jsonl, frames/000000.png, ...}` under it.

## API

`createRecorder(options)` returns a recorder with:

- `startEpisode(seed)`: starts a new episode. Ends the current one first if needed. Calls `actionSource.reset(seed)` if it exists.
- `tick()`: call once per logic step, before the step is applied. Captures the canvas (frame N), asks the action source for action N, records `getState()`, and returns the action for the game to apply. While `state.alive` is false the returned and recorded action is forced to no buttons, as the format requires for game-over frames.
- `endEpisode({corrupted, reason})`: marks the last step line `episode_end: true` and closes the episode. Both options are optional. An episode with zero steps is dropped.
- `save({release})`: waits for PNG encoding to finish and resolves to a zip `Blob` with every finished episode. The episode in progress is not included. With `release: true` the saved episodes are dropped from memory, so later saves only contain newer episodes. All zips from one recorder share the same `session.json` and keep counting episode numbers, so they can be unzipped into the same folder.
- `sessionId`, `recording`, and `pendingFrames` (frames captured but not yet encoded).

Action sources implement `{name, settings, lookScale, reset(seed)?, next(step)}`, where `next` returns `{buttons, look}`:

- `KeyboardActionSource({target, pointerLockElement, lookScale})`: human play (`name: "human"`). Arrows or WASD, Space for jump, Z and X for action_a and action_b. Left and right together cancel out, and so do up and down. With `lookScale` set, clicking `pointerLockElement` locks the pointer, mouse deltas are summed per step, clipped to plus or minus `lookScale` and divided by it. `look[0]` is positive to the right and `look[1]` positive upwards (browser `movementY` is flipped). Call `detach()` to remove its listeners.
- `ReplayActionSource(actions)` or `ReplayActionSource.fromStepsJsonl(text)`: plays back a recorded episode (`name: "replay"`), for determinism tests.
- An autopilot is game-specific; write a class with the same shape and give it a descriptive `name`.

`mulberry32(seed)` is a seeded random generator. `installSeededMathRandom(seed)` swaps it in for `Math.random` and returns a function that puts the original back. Use it so a replay with the same seed and actions gives the same frames.

## Notes

- Frames are downscaled with an `OffscreenCanvas` (smoothing on, quality "high") and saved as PNG. Transparent canvas pixels are saved as black.
- A WebGL canvas can only be read after it has been presented if it was created with `preserveDrawingBuffer: true`, or if `tick()` runs in the same task as the render. Otherwise the frames come out blank.
- Render once per logic step while recording. If the game runs several logic steps per drawn frame, frame N would not be the image the player saw when choosing action N.
- `time` in `steps.jsonl` is `step * logicTimestep`, rounded to 6 decimals.
- Everything is held in memory until `save()`. Memory use grows with every PNG (size depends on the art, not measured yet), so save with `release: true` every few minutes during long autopilot runs.
