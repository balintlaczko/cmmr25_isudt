import torch
from torch import nn
from ..utils.tensor import wrap


class Phasor(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super(Phasor, self).__init__()

        self.sr = sr

    def forward(self, freq: torch.Tensor):  # input shape: (..., n_samples)
        increment = freq[..., :-1] / self.sr
        phase = torch.cumsum(increment, dim=-1)
        phase = torch.cat(
            [torch.zeros_like(freq[..., 0]).unsqueeze(-1).to(phase.device), phase], dim=-1)
        phasor = wrap(phase, 0.0, 1.0)
        return phasor


class Sinewave(nn.Module):
    def __init__(
        self,
        sr: int
    ):
        super(Sinewave, self).__init__()

        self.sr = sr
        self.phasor = Phasor(self.sr)

    def forward(self, freq: torch.Tensor):
        phasor = self.phasor(freq)
        sine = torch.sin(2 * torch.pi * phasor)
        return sine