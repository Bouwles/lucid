# Decisions

One entry per decision not covered by SPEC.md: date, decision, one or two sentences of why.

**2026-11-16. Resolution and frame rate.** Record at 256 on the short side and the game's native logic rate. Train at 64x64 with `frame_skip: 4` (60 Hz logic to 15 Hz training), as SPEC 3.3 suggests. The originals are kept, so both can change later without re-recording.

**2026-11-16. Processed frames are `.npy`, not inside the `.npz`.** Appendix B puts frames in the `.npz`, but `.npz` files can't be memory-mapped. A base-size dataset (1M frames at 64x64) is about 12 GB, so frames go in a separate `.frames.npy` per episode and are memory-mapped. Labels and actions stay in the small `.npz`. See docs/formats.md.

**2026-11-16. Default frame-skip action rule is `first_non_none`.** This is the rule from SPEC 12.2: the first sub-step action that presses anything, otherwise none, so a jump pressed on sub-step 3 isn't lost. `first` and `union` are also implemented and tested. Look is averaged over the interval so it stays in [-1, 1].

**2026-11-16. Hash-based splits.** Each episode's split comes from a hash of `session/episode` plus a seed, not from shuffling. Adding new sessions therefore never moves an old episode between train and test.

**2026-11-16. EDM loss computed on the raw network output.** Since lambda(sigma) times c_out(sigma) squared is 1, an unweighted MSE on F against `(y - c_skip x) / c_out` equals the paper's weighted loss on D. It avoids dividing by tiny numbers at small sigma. `tests/test_edm.py` checks the two agree.

**2026-11-16. Context noise is added at inference too.** At inference the context frames get real Gaussian noise at `inference_context_noise`, and the model is told that level. This keeps inference inputs matching what the model saw in training. Setting it to 0 turns it off.

**2026-11-16. AdaGroupNorm projections use the default init, not zeros.** With zero init the conditioning has no effect at step 0, and the "changing the action changes the output" check can't run on a fresh model.

**2026-11-16. Synthetic lane game for pipeline tests.** `lucid/data/synthetic.py` writes episodes in the raw format so validation, building, training, evaluation and serving can all be tested without Unity. It has its own registry (`configs/games_synthetic.yaml`), so it never mixes with real game ids. It is never used for reported results.

**2026-11-16. Upsampler without a previous frame.** At an episode start there is no previous high-res frame, so the upscaled low-res frame stands in for it. This avoids an extra "has previous frame" input.

**2026-11-16. Latent diffusion is only half built.** The autoencoder and its training script exist (SPEC 14.9 item 8). Training the world model on latents isn't wired in yet. That waits until pixel-space results show it's needed.

**2026-11-16. Null-action bracket for controllability.** SPEC 17.4 compares against a model trained with the action always null. That is a separate training run with `--set model.p_uncond_action=1.0`, so evaluate.py reports it as not measured until that run exists.

**2026-11-16. `protect-raw-data` hook tightened.** The original pattern blocked any command containing "rm" inside a word (like "format") followed later by `data/`. It now only matches `rm`, `mv`, `shred` or `truncate` as whole commands.
