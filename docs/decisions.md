# Decisions

Choices I made along the way, with the reason for each.

**Resolution and frame rate.** Record at 256 on the short side and the game's native logic rate. Train at 64x64, keeping every 4th logic step (60 Hz down to 15 Hz). The originals are kept, so both can change later without recording again.

**Frames are stored as `.npy`, not inside the `.npz`.** A `.npz` can't be memory-mapped, and a full dataset (1M frames at 64x64) is about 12 GB. So frames go in their own `.frames.npy` per episode and are read straight from disk. Labels and actions stay in a small `.npz`.

**Frame-skip action rule: `first_non_none`.** When 4 logic steps become one training step, the kept action is the first one that actually presses something. A jump on the third sub-step still counts. `first` and `union` exist too and are tested. Look is averaged so it stays in [-1, 1].

**Splits come from a hash.** Each episode's split is decided by hashing its session and episode id. Recording more sessions never moves an old episode from train to test.

**EDM loss on the raw network output.** lambda(sigma) times c_out(sigma) squared is exactly 1, so plain MSE on F against `(y - c_skip x) / c_out` is the same as the paper's weighted loss on D. It avoids dividing by tiny numbers at low noise. A test checks the two agree.

**Context noise at play time too.** When playing, the context frames get real noise at the inference level and the model is told that level. That keeps play-time inputs looking like training inputs. Setting it to 0 turns it off.

**AdaGroupNorm starts with normal weights, not zeros.** With zero init the action has no effect at step 0, so the "different actions give different frames" check can't run on a fresh model.

**A synthetic lane game for testing.** `lucid/data/synthetic.py` writes fake episodes in the same raw format, so I can test validation, dataset building, training, evaluation and the server without opening Unity. It has its own game list (`configs/games_synthetic.yaml`), never mixes with my real games, and none of its numbers count as results.

**Upsampler at the start of an episode.** There's no previous high-res frame yet, so the upscaled low-res frame stands in for it.

**Latent diffusion is half built.** The autoencoder and its training script exist. Training the world model on latents waits until pixel-space results show I need it.

**Null-action comparison.** To show how much control the model learned, I'll compare against a copy trained with the action always hidden (`--set model.p_uncond_action=1.0`). Until that run exists, the report says "not measured".

**The desktop app before the games.** The app spec builds Drift Kit first, but Unity isn't installed on my Mac yet, so I'm building the app shell first. The test track's dream is a placeholder until a real game and its dream exist.

**Svelte for the app's interface.** It's small, needs little boilerplate and compiles away, which suits a lightweight Tauri app.

**Dreams on ONNX Runtime in the base app.** The engine never imports PyTorch, so the installer stays small. The ONNX sampler matches the PyTorch one (there's a test for it), and on my Mac's CPU it ran the test track at about 32 FPS, against about 16 through the PyTorch server.

**The test track ships inside the app.** Its dream pack (12 MB) is bundled, so the app can play a dream with no download, and smoke tests have a model to run. Real game dreams will be downloads from GitHub Releases, checked against a SHA-256 in the catalogue.

**Downloads only from LUCID's own releases.** The engine refuses any download URL that isn't under the LUCID GitHub Releases, and rejects any zip with paths that would escape its folder.

**The built-in games are web games, not Unity builds.** The app spec says Unity, but Unity isn't installed here. Drift Kit, Bramble and Outpost are written in TypeScript with Three.js and run inside the app, so they need no separate download. They follow the same raw recording format (`docs/formats.md`), so everything after recording is unchanged. Art and sound are Kenney's CC0 starter kits (`games/CREDITS.md`).

**Bramble is 2D gameplay drawn with 3D models.** The CC0 platformer kit is 3D, so the game moves only along x and y with a side-on camera and parallax backdrops.

**Why the first dreams ignored the controls.** The conditioning vector is a sum of embeddings. At initialisation the game-id embedding had a size of about 11.7, the action embedding about 1, and the difference between "left" and "right" about 0.12. In the trained test-track model, switching left to right changed the output by about 0.0001, while a fresh noise draw changed it by 0.01 to 0.07. The fix: normalise each source before summing (`cond_norm`) and also feed the actions in as image channels (`action_channels`). Weighting the loss toward moving pixels didn't help and is off.

**Dreams use guidance 2.0 (3.0 for Drift Kit) with 3 sampling steps.** In the current Bramble dream, guidance takes "the IDM sees left when left is held" from 29% to 59% (`runs/20261120_124303_bramble_v2/controls_bramble.json`). Guidance doubles the work per step. Two steps would be faster, but they turn these dreams into noise, so the packs use 3 and run at about 9 FPS on this Mac's CPU. Core ML failed on this model in ONNX Runtime, so dreams run on the CPU.

**Training budget.** Each game dream gets about 30 minutes on the M5 (single run, well under the 2-hour limit), so these are first versions. Longer training is the obvious next step.

**The three dreams ship inside the app.** The spec has dreams download on first use. The packs are about 50 MB each, which is fine for a game app, and bundling means everything works offline straight after installing. The build fetches them from GitHub Releases with checksums (`app/src-tauri/resources/models/packs.json`).

**The test track is retired.** It was only a pipeline check and its dream didn't respond to keys, so it's gone from the app. Its files are kept outside the repo.

**Pick dreams by looking at them, not only by the control score.** The inverse dynamics model can be fooled: dreams that melted into mush scored higher on some keys than sharp ones. Each candidate dream now also gets a 40-step frame strip with recorded keys, plus contrast and sharpness against real frames over time. Drift Kit and Outpost keep their first models because the retrained ones melted. Bramble uses the retrained one.

**One dream per game, not one model for all three.** The multi-game model (150 minutes on all three games) melted within 5 to 10 steps, so live switching and blending stay off. The engine and app support them, so a better multi-game pack only needs to be dropped in.

**Bramble's "Let it play itself" uses the policy trained inside the dream.** In the real game it got further (mean distance 19.2 over 20 episodes) than the copy of the autopilot (3.5) and random keys (16.7). Drift Kit and Outpost use the copy.

**The upsampler isn't shipped.** After 45 minutes (2,829 steps) its last validation scored LPIPS 0.193 against 0.142 for bilinear upscaling, and it looked grainier, so sharp mode is disabled in the app.

**The dream-trained Bramble policy saw test footage as starting points.** `scripts/train_in_dream.py` took the dreams' starting frames from the test split as well as training footage. Only the dreams' first frames came from there, and the real-game test plays the game itself with unseen seeds, so the comparison still stands. The script now uses the validation split.
