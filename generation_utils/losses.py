"""Loss and normalization utilities for conditional diffusion training."""

from __future__ import annotations

import torch
import torch.nn as nn


class BregmanCharbonnierLoss(nn.Module):
    """Pixel-domain Bregman divergence induced by the Charbonnier potential."""

    def __init__(self, eta: float = 1e-5) -> None:
        super().__init__()
        self.eta = float(eta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eta_sq = self.eta * self.eta
        rho_target = torch.sqrt(target * target + eta_sq)
        rho_pred = torch.sqrt(pred * pred + eta_sq)
        grad_rho_pred = pred / (rho_pred + 1e-12)
        divergence = rho_target - rho_pred - grad_rho_pred * (target - pred)
        return torch.relu(divergence).mean()


class FrequencyBregmanPureL1Loss(nn.Module):
    """Frequency-domain Bregman divergence for the L1 potential."""

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.fft2(x_pred, norm="ortho")
        fft_true = torch.fft.fft2(x_true, norm="ortho")
        rho_true = fft_true.abs()
        rho_pred = fft_pred.abs()
        xi_pred = torch.sgn(fft_pred)
        linear_term = torch.real(torch.conj(xi_pred) * (fft_true - fft_pred))
        divergence = rho_true - rho_pred - linear_term
        return torch.relu(divergence).mean()


class LinearMinMaxNormalize(nn.Module):
    """Map intensities linearly from [min, max] to [-1, 1]."""

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.denom = max(self.max_val - self.min_val, 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.min_val) / self.denom - 1.0
