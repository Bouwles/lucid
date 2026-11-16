# Concepts

The ideas LUCID uses, in plain English, each with a pointer to the code. References are in [references.md](references.md).

## World model and rollouts

A world model predicts what the game looks like next, given what it looked like recently and what the player is doing. In LUCID the input is the last k frames (the *context*, k = 4 to start) plus the current action, and the output is the next frame.

Feed each prediction back in as context and the model "plays" the game. That loop is an *autoregressive rollout* (`lucid/eval/rollout.py`). A *teacher-forced* rollout cheats by always using real frames as context, which measures single-step accuracy. A *free-running* rollout uses the model's own outputs, which is what playing actually does.

It's the same idea as generating text with a Transformer one token at a time, except each "token" is a whole frame.

## Why it's hard: drift and blur

Every predicted frame is slightly wrong. When it becomes context, the next prediction builds on that error, and after a few seconds a naive model melts into mush. That's *drift*.

A model trained with plain MSE learns the average of all possible futures. If an obstacle could appear left or right, it draws a faint smear in both places. That's *blur*. The baseline in `lucid/models/baseline.py` does exactly this on purpose, as a reference point.

## Diffusion models

A diffusion model learns to remove noise. In training, take the real next frame, add a random amount of Gaussian noise, and train the network to recover the clean frame. In generation, start from pure noise and denoise in a few steps, guided by the context and action. It produces one sharp, plausible future instead of the blurry average.

The noise level is called sigma. Large sigma means nearly pure noise; small sigma means nearly clean.

## EDM

EDM (Karras et al. 2022) is a tidy recipe for diffusion. LUCID uses three parts of it (`lucid/models/edm.py`):

- **Preconditioning.** The raw network F never sees the noisy image directly. The denoiser is `D(x) = c_skip * x + c_out * F(c_in * x)`, with coefficients that depend on sigma. At low noise D mostly copies x (`c_skip` is near 1). At high noise it mostly trusts F. This keeps the network's inputs and outputs at a sensible scale for every sigma, like normalising inputs to a neural net.
- **Training noise levels.** `ln(sigma)` is drawn from a normal distribution, so training concentrates on the noise levels that matter most.
- **Samplers.** `lucid/models/sampler.py` walks sigma from high to low in a handful of Euler (or Heun) steps. A world model needs few steps because the context already pins down most of the answer.

`sigma_data` is the standard deviation of the pixel values. The dataset build measures it and stores it in the manifest.

## Context-noise augmentation (the drift fix)

In training the context frames are perfect. In play they are the model's own slightly wrong outputs, and that mismatch causes drift. The fix (from GameNGen) is to add random noise to the context frames during training and tell the model how much was added. The model learns to clean up imperfect context instead of amplifying the errors. At play time a small fixed amount is used. `configs/train/small_no_ctx_noise.yaml` is the ablation that switches it off.

## Conditioning: how the network "hears" the action

Everything that isn't an image gets turned into one vector c: the action (an MLP over the 8 button bits plus Fourier features of the look direction), the game id (a learned embedding), and the noise levels. c then adjusts every residual block through **adaptive GroupNorm**: after normalising the features, a small linear layer predicts a scale and a shift from c. It's like FiLM layers. The action literally rescales the features everywhere in the network (`lucid/models/blocks.py`).

## Classifier-free guidance (CFG)

During training, the action is replaced with a learned "null action" 10% of the time. At play time the model runs twice, with and without the action, and pushes the result further in the direction the action points: `D_uncond + w * (D_cond - D_uncond)`. This can strengthen weak controls at double the cost. It's off unless `sampler.cfg_scale` is set.

## EMA

The training loop keeps an exponential moving average of the weights, roughly a running average of the last few thousand updates. Generation uses the EMA weights because they're smoother and usually produce cleaner samples than the latest raw weights (`lucid/train/ema.py`).

## Metrics

- **PSNR and SSIM** compare pixels to the real frame. They reward blur, so they can mislead.
- **LPIPS** compares features from a pretrained image network and matches human judgement better. Lower is better.
- **Survival time** counts rollout steps until sharpness or colour statistics leave the range seen in real frames (`lucid/eval/stability.py`). It needs no ground truth, which matters because a dream and the real game diverge after a few seconds anyway.
- **Probes** are small CNNs trained on real frames to read the lane, jump state and alive/dead. Run on dreamed frames, they turn "does the dream make sense?" and "does it obey the controls?" into numbers (`lucid/eval/probes.py`, `lucid/eval/control.py`).

## Frame skip

Unity runs game logic at about 60 steps a second. Training uses every 4th step (15 a second), so each step has more visible motion and rollouts cover more game time. The action for a kept step summarises the 4 logic steps it stands for (`summarise_actions` in `lucid/data/build.py`).

## Multi-game models and blending

One network learns several games by also getting a game embedding. Switching the id mid-rollout should morph the dream into the other game. Mixing two embeddings (50% runner, 50% platformer) is the blending experiment. The engine's `set_blend` passes soft weights instead of an id.

## Two-stage generation: the upsampler

The dynamics model stays small and fast at 64x64, where the game logic lives. A second diffusion model turns each small frame into a sharp 256x256 one. It's conditioned on the small frame and the previous large frame, which stops flicker (`lucid/models/upsampler.py`). It never sees the action, so it can't change what happens, only how crisp it looks.
