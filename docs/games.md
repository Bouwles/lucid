# The built-in games

LUCID ships with three games of its own: **Drift Kit** (low-poly racer), **Bramble** (side-scrolling platformer) and **Outpost** (first-person arena shooter). They're proper games to play, and they're also LUCID's training data, so every choice about how they look and behave affects how good their dreams can be.

They're written in TypeScript and run inside the desktop app (Three.js for rendering). The art and sound come from Kenney's CC0 starter kits. Every asset is credited in [games/CREDITS.md](../games/CREDITS.md).

## Layout

```
games/
  shared/types.ts     the LucidGame contract (below)
  shared/rng.ts       seeded randomness (mulberry32)
  shared/input.ts     keyboard and mouse to unified actions
  shared/runner.ts    fixed-timestep loop for normal play
  registry.ts         id -> lazy import of each game
  <id>/index.ts       exports createGame(): LucidGame
  <id>/assets/        models, textures and sounds (CC0), with their licence files
  CREDITS.md          every asset pack, its licence and link
```

Try a game in a browser while developing: `cd app && npm run dev`, then open `http://localhost:1420/gamedev.html?game=drift-kit`. Add `&mode=recording` to see what LUCID trains on, and `&autopilot=1` to watch the autopilot play.

## The contract

Every game implements `LucidGame` from `games/shared/types.ts`:

- **Fixed logic at 30 ticks a second.** `step(action)` advances exactly one tick. Rendering is separate (`render(alpha)` interpolates between ticks for smooth play).
- **Deterministic.** All randomness goes through the seeded generator passed to `reset(seed)`. No `Math.random`, no wall-clock time, no frame-rate-dependent physics. The same seed and the same actions give the same states.
- **Unified actions.** Buttons are the 8-bit mask from [formats.md](formats.md), plus a look vector for mouse look. Holding a key keeps having an effect every tick (steer, run, move). That makes the controls easy for a dream to learn.
- **Timing.** Frame N is what the player saw when choosing action N, and the action's effect first shows in frame N+1.
- **Episodes.** An episode ends on a crash, a death, a fall or a time limit. The game then shows a recognisable game-over view for 30 ticks with `alive: false`, and `isOver()` becomes true.
- **Labels.** `state()` reports position, vertical state, alive, speed, progress and any extra numbers (heading, health) for evaluation probes. They are never model inputs.
- **Two render modes.** `normal` is what people play: effects, particles, camera shake and sound. `recording` is calm and clean for LUCID: a square view, no particles or shake, no random visual noise, fixed exposure. It still looks like the same game. Neither mode draws text into the canvas; the HUD is drawn by the app, from `hud()`.
- **Autopilot.** `createAutopilot(seed)` returns a player that mixes competent play, sloppy play with mistakes, random held inputs and idling, so the recorded data covers every action, every situation and plenty of deaths.

## Recording

`app/record.html` runs a game in recording mode with its autopilot, as fast as the machine allows, and streams frames, actions and labels to `scripts/record_web_game.py`. That script writes them in the standard raw format, so the dataset builder and everything after it work unchanged. Logic runs at 30 Hz, and datasets keep every second tick (15 Hz) with `frame_skip: 2`.
