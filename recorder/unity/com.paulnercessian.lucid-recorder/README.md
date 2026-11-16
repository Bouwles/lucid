# LUCID Recorder

A Unity package that records a game for LUCID. On every logic step (`FixedUpdate`) it saves the frame from a recording camera, the action that was pressed and a few state labels, in the raw format described in [docs/formats.md](../../../docs/formats.md). Episodes run back to back, each with its own seed, so a human or an autopilot can record for hours.

Unity 2021.3 or newer. It is written for the built-in render pipeline and URP (it relies on `Camera.Render()`), but has not been tested in a Unity project yet.

## Install

In Unity, open Window > Package Manager, click +, choose "Add package from git URL" and paste:

```
https://github.com/Bouwles/lucid.git?path=/recorder/unity/com.paulnercessian.lucid-recorder
```

To try the runner example, select the package in the Package Manager and import "Runner Adapter Example" from the Samples tab.

## What a game implements

A game needs two small pieces of code. Both live in the recording scene, not in the normal game.

**A game adapter**, a `MonoBehaviour` that implements `IGameAdapter` and `IEpisodeControl`:

```csharp
public interface IGameAdapter
{
    void Apply(UnifiedAction action);   // press the game's controls for this step
    GameStateSnapshot ReadState();      // lane or x/y, vertical state, alive, speed, distance
    string GameId { get; }              // id from configs/games.yaml
}

public interface IEpisodeControl
{
    void BeginEpisode(int seed);        // seed the game's RNG and reset to the start of a run
    bool EpisodeReady { get; }          // true once the new run is on screen
}
```

**An action source**, which implements `IActionSource` and decides what is pressed each step. Three come with the package: `HumanActionSource` (keyboard, optional mouse look), `ReplayActionSource` (replays a recorded episode) and `AutoPilotBase`. For the autopilot, a game subclasses `AutoPilotBase` and implements one look-ahead method, `DecideAvoidance(state)`, that says what a competent player would press right now. The base class mixes four behaviours: random actions held for a short time, the competent avoider, a noisy avoider that sometimes reacts wrongly or late, and idling. The mix, the hold times and the mistake chance are set in the `RecorderConfig` and saved in every episode's `meta.json`.

`ActionMapper` is the per-game table from the game's control names ("slide", "fire") to the eight unified buttons, plus the look scale for mouse-look games.

## Quick start (about 10 minutes)

1. Install the package (above).
2. Add `LUCID_RECORDING` to Project Settings > Player > Scripting Define Symbols. All game-side recording code goes behind this define, so normal builds are unchanged. In a player build without it, the recorder disables itself.
3. Duplicate your game scene as a recording scene, for example `Runner_Recording`.
4. Create the assets: Assets > Create > LUCID > Action Mapper, and Assets > Create > LUCID > Recorder Config. In the config, set the output folder to the `data/raw` folder of your LUCID checkout, the resolution (keep the game's aspect ratio, short side 256 or more) and the session length. Assign the action mapper.
5. Import the runner sample, or copy it, and fill in every `TODO` in `RunnerAdapter.cs` and `RunnerAutoPilot.cs` with calls into your game.
6. In the recording scene, add `RunnerAdapter` and `RunnerAutoPilot` (or `HumanActionSource`) to a GameObject. Add a `LucidRecorder` to another GameObject, or let Window > LUCID > Recorder create one.
7. Open Window > LUCID > Recorder, assign the adapter, action source and config, and press "Play and record". The window shows episodes, frames, queue depth and write errors while it runs.
8. Stop with "Stop after this episode" or "Stop now", then check the session from the LUCID repo with `python scripts/validate_raw.py data/raw/<session>`.
9. Open `frames/000000.png` of the first episode once and check it is upright. If it is upside down, set Row Order in the config to Flip and note it in `docs/decisions.md`.

For sessions longer than a few minutes, record in a build with `LUCID_RECORDING` defined and Start On Play ticked on the recorder.

## How recording works

- The recorder runs before every other script in `FixedUpdate`. Each step it renders the recording camera into a RenderTexture (UI layer excluded), starts an `AsyncGPUReadback`, reads the state, asks the action source for an action, and passes it to `adapter.Apply`. So frame N is what the player saw when action N was chosen, and the effect of action N first shows in frame N+1.
- Readbacks are handed to a background writer thread in step order. It encodes PNGs and appends to `steps.jsonl`. The queue is bounded: at 75% full the recorder pauses game logic (through `Time.timeScale`) until it drains to 50%. If it ever fills completely, the game thread waits. Frames are never dropped.
- On death the recorder keeps recording the game-over frame for `gameOverHoldSeconds` with `alive: false` and no buttons, marks the last line `episode_end: true`, writes `meta.json`, and calls `BeginEpisode` with a new seed. Logic stays paused from the reset until step 0 is recorded.
- Any write error marks the episode `corrupted: true` in `meta.json` with the reason. Stopping, quitting and leaving play mode wait for the queue to finish and close every file.

## Determinism

The determinism test (Window > LUCID > Determinism Test) replays an episode with its seed and actions and reports the first step where the state differs, and optionally the first frame whose pixels differ by more than a tolerance. For a replay to match, the recording scene needs:

- **A fixed timestep.** All gameplay and physics run in `FixedUpdate` with a constant `Time.fixedDeltaTime`. Nothing that affects gameplay may use `Time.deltaTime` in `Update`.
- **One seeded RNG.** Every gameplay random choice goes through one generator seeded in `BeginEpisode`. `UnityEngine.Random` is also used by other code, so a private `System.Random` is safer.
- **Input only from `Apply`.** The game must not read the keyboard itself while recording.
- **No self-restart.** The game must not restart on its own after death; the recorder does it.
- **Frames that depend only on game state.** Things that move per render frame make the same state look different from run to run: a camera that follows the player in `LateUpdate` (move it in `FixedUpdate` instead), Animators in Normal update mode (use Animate Physics), Rigidbody interpolation, particle systems, camera shake, motion blur, TAA and other temporal effects. Turn them off or tie them to the logic step for recording.
- **The same machine and settings.** GPUs and drivers can round differently, so pixel-identical frames are only expected on the same machine, graphics API and quality settings. Across machines, compare with a small tolerance and rely on the state check.

## Notes

- The recording camera is a copy of the main camera (or one you assign). With URP the copy gets default camera data, so post-processing is off unless you set it up on your own recording camera.
- Colour: the RenderTexture is sRGB, so PNGs hold the same colours as the screen in both Linear and Gamma projects.
- The package writes only the raw format. Building the training dataset happens in Python (`scripts/build_dataset.py`).
