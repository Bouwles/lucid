"""Write synthetic episodes in the raw recording format, for pipeline tests only.

The synthetic lane game (lucid/data/synthetic.py) is not one of Paul's games and is never used
for reported results. It lets every stage after recording be tested without Unity.
"""

import argparse

from lucid.data.synthetic import record_session
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--size", type=int, default=128, help="recording resolution (square)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--human-every", type=int, default=10, help="label every n-th episode as human play"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    path = record_session(
        args.out, args.episodes, args.seed, args.size, args.max_steps, args.human_every
    )
    print(f"wrote {args.episodes} synthetic episodes to {path}")


if __name__ == "__main__":
    main()
