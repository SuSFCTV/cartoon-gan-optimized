import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# Добавим в sys.path путь до third_party/cartoon-gan, чтобы импортировать Generator
ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT / "third_party" / "cartoon-gan"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Импортируем класс генератора из оригинального репо
# Проверь путь: в исходнике обычно models/generator.py, класс может называться Generator
from models.generator import Generator  # noqa: E402

# Базовые трансформы: как у автора - ToTensor() c [0,1]; размер выберем 512x512 по умолчанию
def get_transforms(size: int = 512) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),  # [0,1] float32
    ])

def load_image_as_tensor(path: str, size: int = 512) -> torch.Tensor:
    tfm = get_transforms(size)
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1,3,H,W)
    return x

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: (1,3,H,W), [0,1], clamp
    x = x.detach().cpu().clamp(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(x.squeeze(0))

def save_image(x: torch.Tensor, out_path: str):
    img = tensor_to_pil(x)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in exts:
            files.append(str(p))
    return sorted(files)

def load_fp32_model(weights_path: str) -> nn.Module:
    model = Generator()
    state = torch.load(weights_path, map_path=None, map_location="cpu")
    # Для совместимости с разными сохранениями:
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

def to_channels_last(model: nn.Module) -> nn.Module:
    # Для ускорения на CPU/OneDNN можно хранить тензоры в channels_last
    for p in model.parameters():
        p.data = p.data.contiguous(memory_format=torch.channels_last)
    return model