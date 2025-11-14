import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping, prepare_fx, convert_fx

from utils import load_fp32_model, load_image_as_tensor, list_images

torch.backends.quantized.engine = "fbgemm"
torch.set_grad_enabled(False)

def calibrate(prepared: nn.Module, calib_dir: str, size: int = 512, max_imgs: int = 64):
    paths = list_images(calib_dir)
    if not paths:
        raise RuntimeError(f"No images in calib_dir: {calib_dir}")
    paths = paths[:max_imgs]
    for p in paths:
        x = load_image_as_tensor(p, size=size)
        _ = prepared(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parents[2] / "third_party/cartoon-gan/checkpoints/generator.pth"))
    ap.add_argument("--calib_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data/calib"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out_pt", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_int8_fx.pt"))
    ap.add_argument("--also_save_fp32_ts", action="store_true")
    args = ap.parse_args()

    # 1) Загружаем FP32 модель
    fp32 = load_fp32_model(args.weights)
    fp32.eval()

    # (Опционально) TorchScript FP32 — контрольная точка для мобильного/сравнения
    if args.also_save_fp32_ts:
        ts = torch.jit.trace(fp32, torch.randn(1,3,args.size,args.size))
        ts.save(str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_fp32.ts"))
        print("Saved FP32 TorchScript to checkpoints/cartoon_gan_fp32.ts")

    # 2) FX PTQ
    example_input = torch.randn(1, 3, args.size, args.size)
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")  # x86 CPU

    print("Preparing FX graph for quantization...")
    prepared = prepare_fx(fp32, qconfig_mapping, example_input)

    print("Calibrating on images:", args.calib_dir)
    calibrate(prepared, calib_dir=args.calib_dir, size=args.size, max_imgs=64)

    print("Converting to INT8...")
    quantized = convert_fx(prepared)
    quantized.eval()

    # 3) Сохраняем целиком модуль (для десктоп-инференса)
    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, str(out_path))
    print(f"Saved INT8 quantized model: {out_path}")

if __name__ == "__main__":
    main()