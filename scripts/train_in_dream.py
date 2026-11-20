"""Train a policy only inside the dream, starting from the behaviour-cloning policy.

python scripts/train_in_dream.py --world runs/<dream run>/latest.pt --game bramble --policy <policy.pt> \
    --dataset bramble_v1 --minutes 30 --out runs/dream_rl_bramble

The reward model learns from real footage. Everything the policy experiences during training is
dreamed. Test the result in the real game with scripts/eval_policy_real.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import load_episodes
from lucid.data.format import BUTTONS
from lucid.dream_rl import train_in_dream, train_reward
from lucid.eval.rollout import sampler_kwargs
from lucid.models.factory import load_model
from lucid.policy import Policy, export_policy
from lucid.utils.config import load_games
from lucid.utils.device import pick_device
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--world", required=True, help="dream model checkpoint")
    p.add_argument("--game", required=True)
    p.add_argument("--policy", required=True, help="behaviour-cloning policy.pt to start from")
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default="data/processed")
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    seed_everything(0)
    device = pick_device(args.device)
    world, cfg, extra = load_model(args.world, device)
    game_id = (extra.get("games") or {})[args.game]
    games = load_games(cfg.games_file)
    used = sorted({BUTTONS.index(b) for b in games[args.game]["actions"].values()})

    train_eps, _ = load_episodes(args.root, args.dataset, "train", [args.game])
    print("training the reward model on real frames")
    reward, scale = train_reward(train_eps, device)

    test_eps, _ = load_episodes(args.root, args.dataset, "test", [args.game])
    k = world.k
    rng = np.random.default_rng(0)
    frames, buttons, look = [], [], []
    for e in test_eps + train_eps[:20]:
        for t in range(0, len(e) - k, 25):
            if bool(e.labels["alive"][t : t + k].all()):
                frames.append(np.asarray(e.frames[t : t + k]))
                buttons.append(e.buttons[t : t + k])
                look.append(e.look[t : t + k])
    order = rng.permutation(len(frames))[:400]
    starts = {
        "frames": np.stack(frames)[order],
        "buttons": np.stack(buttons)[order],
        "look": np.stack(look)[order],
    }

    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    pol = Policy(ck["k"]).to(device)
    pol.load_state_dict(ck["state"])
    pol.train()
    gk = sampler_kwargs(cfg.sampler, cfg_scale=2.0)
    print(f"training inside the dream for {args.minutes} minutes with {args.envs} parallel dreams")
    hist = train_in_dream(
        world,
        pol,
        reward,
        starts,
        device,
        used,
        game_id,
        minutes=args.minutes,
        envs=args.envs,
        gen_kwargs=gk,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": pol.state_dict(), "k": pol.k, "trained": "inside the dream"}, out / "policy.pt"
    )
    W, H = cfg.data.resolution
    export_policy(pol, out / "policy.onnx", pol.k, H, W)
    (out / "dream_training.json").write_text(
        json.dumps({"progress_scale": scale, **hist}, indent=1)
    )
    r = hist["returns"]
    print(
        f"done: {hist['iterations']} iterations; dreamed return {np.mean(r[:10]):.2f} at the start, {np.mean(r[-10:]):.2f} at the end"
    )


if __name__ == "__main__":
    main()
