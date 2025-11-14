import argparse
from pathlib import Path
import torch
from infer import run_infer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parents[2] / "third_party/cartoon-gan/checkpoints/generator.pth"))
    ap.add_argument("--test_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data/test"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--int8_model_path", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints/cartoon_gan_int8_fx.pt"))
    args = ap.parse_args()

    # GPU
    if torch.cuda.is_available():
        run_infer(
            weights=args.weights,
            input_path=args.test_dir,
            out_dir=str(Path(__file__).resolve().parents[1] / "outputs/baseline"),
            device="cuda",
            size=args.size
        )
    else:
        print("CUDA not available; skipping GPU.")

    # CPU FP32
    run_infer(
        weights=args.weights,
        input_path=args.test_dir,
        out_dir=str(Path(__file__).resolve().parents[1] / "outputs/baseline"),
        device="cpu",
        size=args.size
    )

    # CPU INT8
    run_infer(
        weights=args.weights,
        input_path=args.test_dir,
        out_dir=str(Path(__file__).resolve().parents[1] / "outputs/int8"),
        device="int8",
        size=args.size,
        int8_model_path=args.int8_model_path
    )

if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()