# src/int8_ptq_inference.py
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.quantization as tq


from .config import IMG_SIZE
from .cartoon_gan_model import load_generator
from .image_utils import load_image_tensor, tensor_to_pil_image


class QuantizedGeneratorWrapper(nn.Module):
    """
    Обёртка над исходным Generator.
    Квантуем только down-блок (Conv2d), res и up оставляем float.
    """
    def __init__(self, base_generator: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.gen = base_generator

    def forward(self, x):
        # вход -> квант
        x = self.quant(x)

        # квантованная часть: down
        x = self.gen.down(x)

        # обратно в float
        x = self.dequant(x)

        # float часть: res + up
        x = self.gen.res(x)
        x = self.gen.up(x)
        return x


def calibrate(model: nn.Module, calib_dir: Path, num_images: int = 200):
    model.eval()
    with torch.inference_mode():
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(list(calib_dir.glob(ext)))
        img_paths = img_paths[:num_images]

        if not img_paths:
            print("[INT8] В calib_dir нет картинок, используем случайный шум")
            for _ in range(num_images):
                x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                x = x * 2.0 - 1.0
                _ = model(x)
            return

        print(f"[INT8] Calibration on {len(img_paths)} images...")
        for p in img_paths:
            x = load_image_tensor(p, size=IMG_SIZE)
            x = x * 2.0 - 1.0
            _ = model(x)


def build_int8_model(calib_dir: Path) -> nn.Module:
    device = torch.device("cpu")

    float_gen = load_generator(device=device)
    float_gen.eval()

    model = QuantizedGeneratorWrapper(float_gen)
    model.eval()

    engines = torch.backends.quantized.supported_engines
    print(f"[INT8] Supported quantized engines: {engines}")
    if "onednn" in engines:
        torch.backends.quantized.engine = "onednn"
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    else:
        raise RuntimeError(f"No supported quantized engine: {engines}")

    model.qconfig = tq.default_qconfig

    model.gen.res.qconfig = None
    model.gen.up.qconfig = None

    for name, module in model.gen.down.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.qconfig = None
            print(f"[INT8] Disable quantization for BatchNorm2d at gen.down.{name}")

    print("[INT8] Global qconfig:", model.qconfig)

    print("[INT8] Inserting observers (prepare)...")
    tq.prepare(model, inplace=True)

    print("[INT8] Calibration...")
    calibrate(model, calib_dir=calib_dir, num_images=200)

    print("[INT8] Converting to quantized model (convert)...")
    tq.convert(model, inplace=True)
    model.eval()

    print("[INT8] Example gen.down[0] after quantization:\n", model.gen.down[0])
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Post-training static INT8 quantization + inference for CartoonGAN generator (CPU)"
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        required=True,
        help="Папка с калибровочными изображениями (реальные фото)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Входное изображение",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Куда сохранить результат INT8-инференса",
    )
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_int8 = build_int8_model(calib_dir)

    x = load_image_tensor(input_path, size=IMG_SIZE)
    x = x * 2.0 - 1.0

    with torch.inference_mode():
        for _ in range(3):
            _ = model_int8(x)

        t0 = time.perf_counter()
        y = model_int8(x)
        dt = time.perf_counter() - t0

    y = (y.clamp(-1, 1) + 1.0) / 2.0
    img_out = tensor_to_pil_image(y)
    img_out.save(output_path.as_posix())

    print(f"[INT8] Saved cartoonized image to: {output_path}")
    print(f"[INT8] Inference time (CPU INT8) = {dt * 1000:.1f} ms")


if __name__ == "__main__":
    main()