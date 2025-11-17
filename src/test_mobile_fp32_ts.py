# src/test_mobile_fp32_ts.py
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T


IMG_SIZE = 256


preprocess = T.Compose(
    [
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

to_pil = T.ToPILImage()


def run_ts_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    device: str = "cpu",
):
    dev = torch.device(device)
    print(f"[TS-TEST] Loading TorchScript model from {model_path}")
    model = torch.jit.load(model_path.as_posix(), map_location=dev)
    model.eval()

    img = Image.open(input_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        y = model(x)[0].cpu()

    y = y * 0.5 + 0.5
    y = torch.clamp(y, 0.0, 1.0)

    out_img = to_pil(y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    print(f"[TS-TEST] Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Входное изображение")
    parser.add_argument("-o", "--output", type=str, required=True, help="Выходное изображение")
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/cartoon_gan_mobile_fp32.pt",
        help="Путь к TorchScript FP32 модели",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="cpu или cuda (желательно cpu)"
    )
    args = parser.parse_args()

    run_ts_inference(
        model_path=Path(args.model),
        input_path=Path(args.input),
        output_path=Path(args.output),
        device=args.device,
    )


if __name__ == "__main__":
    main()