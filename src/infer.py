import argparse
import os
from pathlib import Path
import time
import torch

from utils import load_fp32_model, load_image_as_tensor, save_image, list_images

torch.set_grad_enabled(False)

def run_infer(
    weights: str,
    input_path: str,
    out_dir: str,
    device: str = "cuda",
    size: int = 512,
    int8_model_path: str = None
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if device == "int8":
        # Загрузка квантованной модели, сохранённой целиком (convert_fx)
        print(f"[INT8] Loading quantized model: {int8_model_path}")
        model = torch.load(int8_model_path, map_location="cpu")
        model.eval()
        device_torch = torch.device("cpu")
    else:
        print(f"[{device}] Loading FP32 model: {weights}")
        model = load_fp32_model(weights)
        device_torch = torch.device(device)
        model.to(device_torch)
        if device_torch.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True

    def process_one(img_path: str):
        x = load_image_as_tensor(img_path, size=size)
        if device == "cuda":
            x = x.to(device_torch, memory_format=torch.channels_last, non_blocking=True)
        else:
            x = x.to(device_torch, non_blocking=True)

        t0 = time.perf_counter()
        y = model(x)
        if device_torch.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        out_path = os.path.join(out_dir, Path(img_path).stem + f"_{device}.png")
        save_image(y, out_path)
        print(f"Saved: {out_path}  |  {dt*1000:.1f} ms")
        return dt

    paths = []
    if os.path.isdir(input_path):
        paths = list_images(input_path)
    else:
        paths = [input_path]

    times = []
    for p in paths:
        times.append(process_one(p))

    if times:
        print(f"Avg time ({device}): {sum(times)/len(times)*1000:.1f} ms over {len(times)} images")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parents[2] / "third_party/cartoon-gan/checkpoints/generator.pth"))
    ap.add_argument("--input", type=str, required=True, help="file or folder")
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "outputs" / "baseline"))
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "int8"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--int8_model_path", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_int8_fx.pt"))
    args = ap.parse_args()

    run_infer(
        weights=args.weights,
        input_path=args.input,
        out_dir=args.out_dir,
        device=args.device,
        size=args.size,
        int8_model_path=args.int8_model_path
    )