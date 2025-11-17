import torch
from pathlib import Path

from .cartoon_gan_model import load_generator


IMG_SIZE = 256


def main():
    device = torch.device("cpu")
    print(f"[EXPORT-CPU] Using device: {device}")

    gen = load_generator(device)

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        ts_model = torch.jit.trace(gen, example)
    ts_model.eval()
    ts_model.to("cpu")

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cartoon_gan_mobile_fp32_cpu.pt"

    ts_model.save(out_path.as_posix())
    print(f"[EXPORT-CPU] Saved TorchScript CPU model to: {out_path}")

    m = torch.jit.load(out_path.as_posix(), map_location="cpu")
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        y = m(x)
    print("[EXPORT-CPU] Test forward OK, output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
