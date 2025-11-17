import argparse
from pathlib import Path

import torch
import torch.quantization as tq

from .config import ARTIFACTS_DIR, IMG_SIZE
from .quant_generator import create_fp32_quant_wrapper
from .image_utils import load_image_tensor


def calibrate(model: torch.nn.Module, calib_dir: Path, num_images: int = 20):
    model.eval()
    with torch.inference_mode():
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(list(calib_dir.glob(ext)))
        img_paths = img_paths[:num_images]

        if not img_paths:
            print("[INT8] В calib_dir нет картинок, прогоняем случайный шум")
            for _ in range(num_images):
                x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                x = x * 2.0 - 1.0
                _ = model(x)
            return

        for p in img_paths:
            x = load_image_tensor(p, size=IMG_SIZE)
            x = x * 2.0 - 1.0
            _ = model(x)



def main():
    parser = argparse.ArgumentParser(description="Eager static INT8 quantization for CartoonGAN generator")
    parser.add_argument(
        "--calib_dir",
        type=str,
        required=True,
        help="Папка с калибровочными изображениями (реальные фото)",
    )
    parser.add_argument(
        "--ts_out",
        type=str,
        default=str(ARTIFACTS_DIR / "cartoon_gan_int8_eager.pth"),
        help="Куда сохранить INT8-модель (eager nn.Module)",
    )
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    ts_out = Path(args.ts_out)
    ts_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    qwrap_fp32 = create_fp32_quant_wrapper(device)

    torch.backends.quantized.engine = "onednn"
    qwrap_fp32.qconfig = tq.default_qconfig

    qwrap_fp32.gen.res.qconfig = None
    qwrap_fp32.gen.up.qconfig = None

    print("[INT8] qconfig:", qwrap_fp32.qconfig)

    print("[INT8] Inserting observers (prepare)...")
    tq.prepare(qwrap_fp32, inplace=True)

    print("[INT8] Calibration...")
    calibrate(qwrap_fp32, calib_dir=calib_dir, num_images=20)

    print("[INT8] Converting to quantized model...")
    tq.convert(qwrap_fp32, inplace=True)
    qwrap_fp32.eval()
    print("[INT8] Example down-layer after quantization:\n", qwrap_fp32.gen.down[0])

    ts_out = Path(args.ts_out)
    ts_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(qwrap_fp32, ts_out.as_posix())
    print(f"[INT8] Saved eager INT8 model (pickled nn.Module) to: {ts_out}")


if __name__ == "__main__":
    main()
