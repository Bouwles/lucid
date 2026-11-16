# Hardware

Written by `python scripts/check_env.py`. Re-run it on any new machine.

- Date: 2026-11-16
- OS: macOS-26.6-arm64-arm-64bit
- CPU: arm
- Python: 3.12.12
- PyTorch: 2.15.0
- Device: mps
- GPU: Apple Silicon (Apple M5), unified memory 32 GB
- bf16 autocast works: True
- fp16 autocast works: True
- Precision the training loop will use for `bf16`: torch.bfloat16
- fp32 matmul: 3.60 TFLOPS (2048x2048, rough)
- fp16 matmul: 14.63 TFLOPS (2048x2048, rough)

## Measured training times

| Config | Machine | Steps | Wall time | Samples/s |
|---|---|---|---|---|
