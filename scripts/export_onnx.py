"""Export a world model's denoiser to ONNX for the WebGPU client (SPEC 19.1 option B)."""

import argparse

from lucid.models.factory import load_model
from lucid.play.export_onnx import export_denoiser


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="web/model/denoiser.onnx")
    args = p.parse_args()
    model, cfg, _ = load_model(args.checkpoint, "cpu")
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {export_denoiser(model, cfg, args.out)}")


if __name__ == "__main__":
    main()
