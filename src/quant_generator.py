# src/quant_generator.py
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

from .cartoon_gan_model import Generator, load_generator  # load_generator ты уже используешь


class QuantizedGeneratorWrapper(nn.Module):
    """
    Обёртка над CartoonGAN Generator
    INT8 будет только на блоке self.gen.down, всё остальное останется в float.
    """
    def __init__(self, base_generator: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # сам генератор с уже загруженными весами
        self.gen = base_generator

    def forward(self, x):
        # квантуем вход
        x = self.quant(x)

        # КВАНТОВАННАЯ часть (down-encoder)
        x = self.gen.down(x)

        # возвращаемся в float
        x = self.dequant(x)

        # float часть (res-блоки + up-decoder)
        x = self.gen.res(x)
        x = self.gen.up(x)
        return x


def create_fp32_quant_wrapper(device: torch.device) -> QuantizedGeneratorWrapper:
    """
    Загружает обычный FP32 Generator с весами и оборачивает его в QuantizedGeneratorWrapper.
    """
    float_gen = load_generator(device=torch.device("cpu"))
    float_gen.eval()

    qwrap = QuantizedGeneratorWrapper(float_gen)
    qwrap.eval()
    return qwrap
