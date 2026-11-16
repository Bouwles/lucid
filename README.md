<p align="center">
  <img src="docs/media/hero.gif" alt="Real game on the left, LUCID's dream on the right, driven by the same inputs" width="760">
</p>

<h1 align="center">LUCID</h1>

<p align="center"><b>A neural game engine that learns to dream the games I make.</b></p>

<p align="center">
  <a href="#play-it">Play it</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#what-didnt-work">What didn't work</a> ·
  <a href="docs/writeup.md">Write-up</a>
</p>

---

The GIF above is two games. The left one is a game I made in Unity. The right one has no game engine at all: a neural network is looking at the last few frames and the keys being pressed, and imagining the next frame, over and over.

LUCID learns several of my games in one model, an endless runner, a platformer and a 3D game, and you can switch between them while you play.

> **Status:** in development. This README fills in as each stage is finished. See [progress](#progress).

## Play it

**[Play LUCID in your browser](#)** *(link goes live at stage 1)*

Arrow keys to move, space to jump. Pick a game from the menu. Everything you see is being generated frame by frame.

## How it works

```
  last 4 frames --+
  your keys       +-->  diffusion model  -->  next frame --+
  which game    --+                                        |
       ^                                                   |
       +------------------ fed back in --------------------+
```

1. I added a recorder to my Unity games that saves every frame along with the keys pressed. A scripted autopilot plays for hours so I don't have to (plus some of my own play, which it can't fake).
2. A diffusion model learns to predict the next frame from the previous ones, the current input and which game it is.
3. To play, the model's own output is fed back in as the next input. The hard part is stopping small mistakes from snowballing into mush after a few seconds. The trick that helped most is in the [write-up](docs/writeup.md).
4. A second model upscales the small frames to something sharp.

## Results

*Filled in as experiments finish. Numbers come from [docs/results.md](docs/results.md), which also has the methodology.*

| | Runner | Platformer | 3D game |
|---|---|---|---|
| Responds correctly to controls | – | – | – |
| Frames before it falls apart (median) | – | – | – |
| People who guessed the real one (blind test) | – | – | – |
| Frames per second in the browser | – | – | – |

## What didn't work

*The honest list. Updated as I go.*

## Progress

- [ ] Stage 1: playable dream of the runner
- [ ] Stage 2: one model, two games, live switching
- [ ] Stage 3: a 3D game
- [ ] Stage 4: high-resolution output
- [ ] Stage 5: game blending and dream mode
- [ ] Stage 6: an agent that learns to play only inside the dream, tested in the real game

## Use the recorder in your own Unity game

The recorder is a drop-in Unity package. Add it from git, implement two small interfaces for your game, and it records LUCID training data. Instructions: [recorder/unity](recorder/unity/).

## Reproduce it

```bash
git clone https://github.com/Bouwles/lucid && cd lucid
pip install -e .
python scripts/build_dataset.py --config configs/data/runner_v1.yaml
python scripts/train.py --config configs/train/small.yaml
python scripts/evaluate.py --checkpoint runs/<run>/best.pt
python scripts/serve.py --checkpoint runs/<run>/best.pt
```

Full details for every stage are in [SPEC.md](SPEC.md).

## How this was built

The design decisions, experiments and mistakes are logged in [docs/devlog.md](docs/devlog.md) and [docs/decisions.md](docs/decisions.md).

It builds on research I learned a lot from: GameNGen (diffusion models as real-time game engines), DIAMOND (diffusion world models), EDM (the diffusion formulation used here), Genie, and Ha and Schmidhuber's World Models. Links are in [SPEC.md, Appendix D](SPEC.md#appendix-d-references).

All training data comes from games I made. No other games or assets were used.

---

<p align="center">Made by Paul Nercessian · <a href="https://github.com/Bouwles">github.com/Bouwles</a></p>
