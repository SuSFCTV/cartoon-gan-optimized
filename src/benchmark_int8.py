import argparse
import statistics
import time
from pathlib import Path

import torch

from .config import ARTIFACTS_DIR, IMG_SIZE


def benchmark_int8(
    ts_model_path: str | None = None,
    runs: int = 30,
    warmup: int = 10,
) -> None:
    if ts_model_path is None:
        ts_model_path = ARTIFACTS_DIR / "cartoon_gan_int8_ts.pt"
    ts_model_path = Path(ts_model_path)

    if not ts_model_path.exists():
        raise FileNotFoundError(
            f"Quantized TorchScript model not found: {ts_model_path}. "
            f"Run quantize_int8.py first."
        )

    print(f"[INT8 Benchmark] Loading quantized TorchScript model from {ts_model_path}")
    model = torch.jit.load(ts_model_path.as_posix(), map_location="cpu")
    model.eval()

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    x = x * 2.0 - 1.0

    print(f"[INT8 Benchmark] device=cpu, img_size={IMG_SIZE}, runs={runs}, warmup={warmup}")

    # прогрев
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            dt = time.perf_counter() - t0
            times.append(dt)

    mean_ms = statistics.mean(times) * 1000
    p50_ms = statistics.median(times) * 1000
    p90_ms = statistics.quantiles(times, n=10)[8] * 1000

    print(f"[INT8 cpu] mean={mean_ms:.2f} ms, p50={p50_ms:.2f} ms, p90={p90_ms:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT8 CartoonGAN on CPU")
    parser.add_argument("--ts_model", type=str, default=None)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    benchmark_int8(args.ts_model, runs=args.runs, warmup=args.warmup)


if __name__ == "__main__":
    main()
