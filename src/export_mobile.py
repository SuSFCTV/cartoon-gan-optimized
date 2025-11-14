import argparse
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8_model_path", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_int8_fx.pt"))
    ap.add_argument("--out_ptl", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_int8_mobile.ptl"))
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading INT8 PyTorch model: {args.int8_model_path}")
    model = torch.load(args.int8_model_path, map_location="cpu")
    model.eval()

    # Для TorchMobile нужен TorchScript
    print("Scripting...")
    scripted = torch.jit.script(model)

    out_path = Path(args.out_ptl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Saving for Lite interpreter...")
    scripted._save_for_lite_interpreter(str(out_path))
    print(f"Saved mobile model: {out_path}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()