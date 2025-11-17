import argparse
from pathlib import Path
import time
import torch

from .config import GPU_DEVICE, CPU_DEVICE
from .cartoon_gan_model import load_generator
from .image_utils import load_image_tensor, tensor_to_pil_image


def run_inference(input_path: str, output_path: str, device_str: str) -> None:
    device = torch.device(device_str)
    model = load_generator(device)
    print(f"[FP32] Using device: {device}")

    x = load_image_tensor(input_path)
    x = x * 2.0 - 1.0
    x = x.to(device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    y = (y + 1.0) / 2.0
    img_out = tensor_to_pil_image(y)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img_out.save(output_path)

    print(f"[FP32] Saved to {output_path}, time = {dt * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="FP32 CartoonGAN inference (CPU/GPU)")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input image path")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output image path")
    parser.add_argument(
        "-d", "--device", type=str, default=GPU_DEVICE,
        choices=[GPU_DEVICE, CPU_DEVICE],
        help="Device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()
    run_inference(args.input, args.output, args.device)


if __name__ == "__main__":
    main()
