import argparse
import statistics
import time
import torch

from .config import GPU_DEVICE, CPU_DEVICE, IMG_SIZE
from .cartoon_gan_model import load_generator


def benchmark(device_str: str, runs: int = 30, warmup: int = 10) -> None:
    device = torch.device(device_str)
    model = load_generator(device)
    model.eval()

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    x = x * 2.0 - 1.0
    x = x.to(device)

    print(f"[FP32 Benchmark] device={device}, img_size={IMG_SIZE}, runs={runs}, warmup={warmup}")

    # прогрев
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

    mean_ms = statistics.mean(times) * 1000
    p50_ms = statistics.median(times) * 1000
    p90_ms = statistics.quantiles(times, n=10)[8] * 1000

    print(f"[FP32 {device.type}] mean={mean_ms:.2f} ms, p50={p50_ms:.2f} ms, p90={p90_ms:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CartoonGAN FP32 (CPU/GPU)")
    parser.add_argument("-d", "--device", type=str, default=GPU_DEVICE,
                        choices=[GPU_DEVICE, CPU_DEVICE])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    benchmark(args.device, runs=args.runs, warmup=args.warmup)


if __name__ == "__main__":
    main()
