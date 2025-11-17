# src/dump_int8_weights.py
import argparse
from pathlib import Path

import torch
import torch.nn.quantized as nnq

from .int8_ptq_inference import build_int8_model
from .config import IMG_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Build INT8 CartoonGAN (PTQ) and dump its quantized weights"
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        required=True,
        help="Папка с калибровочными изображениями (как для int8_ptq_inference)",
    )
    parser.add_argument(
        "--state_dict_out",
        type=str,
        default="artifacts/cartoon_gan_int8_state_dict.pth",
        help="Куда сохранить state_dict квантованной модели",
    )
    parser.add_argument(
        "--weights_out",
        type=str,
        default="artifacts/cartoon_gan_int8_quant_params.pth",
        help="Куда сохранить подробные INT8 веса (int_repr + scale + zero_point + bias)",
    )
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    state_dict_out = Path(args.state_dict_out)
    weights_out = Path(args.weights_out)

    state_dict_out.parent.mkdir(parents=True, exist_ok=True)
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    print("[INT8-DUMP] Building INT8 model via PTQ...")
    model_int8 = build_int8_model(calib_dir)

    sd = model_int8.state_dict()
    torch.save(sd, state_dict_out.as_posix())
    print(f"[INT8-DUMP] Saved full INT8 state_dict to: {state_dict_out}")

    quant_params = {}
    for name, module in model_int8.named_modules():
        if isinstance(module, nnq.Conv2d):
            w_q = module.weight()
            w_int8 = w_q.int_repr()
            scale = module.scale
            zero_point = module.zero_point
            bias = module.bias()

            base_key = f"{name}"

            quant_params[f"{base_key}.weight_int8"] = w_int8.cpu()
            quant_params[f"{base_key}.scale"] = torch.tensor(scale)
            quant_params[f"{base_key}.zero_point"] = torch.tensor(zero_point)
            if bias is not None:
                quant_params[f"{base_key}.bias"] = bias.cpu()

            print(f"[INT8-DUMP] Found quantized conv: {name} "
                  f"(shape={tuple(w_int8.shape)}, scale={scale}, zp={zero_point})")

    torch.save(quant_params, weights_out.as_posix())
    print(f"[INT8-DUMP] Saved detailed INT8 weights to: {weights_out}")

    print("\n[INT8-DUMP] Done.")
    print("  state_dict:   ", state_dict_out)
    print("  quant_params: ", weights_out)


if __name__ == "__main__":
    main()