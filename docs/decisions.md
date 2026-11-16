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
