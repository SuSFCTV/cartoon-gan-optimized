# src/test_int8_from_state_dict.py
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.quantization as tq

from .config import IMG_SIZE
from .cartoon_gan_model import load_generator
from .image_utils import load_image_tensor, tensor_to_pil_image
from .int8_ptq_inference import QuantizedGeneratorWrapper


def build_quantized_skeleton() -> nn.Module:
    device = torch.device("cpu")

    float_gen = load_generator(device=device)
    float_gen.eval()

    model = QuantizedGeneratorWrapper(float_gen)
    model.eval()

    engines = torch.backends.quantized.supported_engines
    print(f"[TEST-INT8] Supported quantized engines: {engines}")
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

    print("[TEST-INT8] qconfig:", model.qconfig)

    tq.prepare(model, inplace=True)

    with torch.inference_mode():
        x_dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        x_dummy = x_dummy * 2.0 - 1.0
        _ = model(x_dummy)

    tq.convert(model, inplace=True)
    model.eval()

    print("[TEST-INT8] Quantized skeleton built.")
    print("[TEST-INT8] Example down[0]:", model.gen.down[0])

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Test loading INT8 CartoonGAN from saved state_dict and run inference"
    )
    parser.add_argument(
        "--state_dict",
        type=str,
        default="artifacts/cartoon_gan_int8_state_dict.pth",
        help="Путь к сохранённому state_dict квантованной модели",
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
        help="Куда сохранить результат инференса INT8-модели из state_dict",
    )
    args = parser.parse_args()

    state_dict_path = Path(args.state_dict)
    if not state_dict_path.exists():
        raise FileNotFoundError(f"state_dict not found: {state_dict_path}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_int8 = build_quantized_skeleton()

    print(f"[TEST-INT8] Loading state_dict from {state_dict_path}")
    sd = torch.load(state_dict_path.as_posix(), map_location="cpu")
    missing, unexpected = model_int8.load_state_dict(sd, strict=False)
    print("[TEST-INT8] load_state_dict: missing keys =", missing)
    print("[TEST-INT8] load_state_dict: unexpected keys =", unexpected)

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

    print(f"[TEST-INT8] Saved output to: {output_path}")
    print(f"[TEST-INT8] Inference time (CPU INT8, from state_dict) = {dt * 1000:.1f} ms")


if __name__ == "__main__":
    main()
