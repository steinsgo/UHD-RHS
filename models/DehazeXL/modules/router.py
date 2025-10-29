from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RoutingDecision:
    expert_ids: torch.Tensor
    confidences: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor


class InstanceRouter(nn.Module):
    """Top-1 instance-level router with optional temperature scaling."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts, bias=True),
        )
        self._temperature = float(max(init_temperature, 1e-3))

    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        self._temperature = float(max(temperature, 1e-3))

    def forward(
        self,
        descriptor: torch.Tensor,
        temperature: Optional[float] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.net(descriptor)
        temp = temperature if temperature is not None else self._temperature
        scaled_logits = logits / max(temp, 1e-3)
        probs = F.softmax(scaled_logits, dim=-1)
        if return_logits:
            return probs, logits
        return probs, None

    def route(
        self,
        descriptor: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> RoutingDecision:
        probs, logits = self.forward(descriptor, temperature=temperature, return_logits=True)
        confidences, expert_ids = probs.max(dim=-1)
        return RoutingDecision(expert_ids=expert_ids, confidences=confidences, probabilities=probs, logits=logits)

    @staticmethod
    def routing_loss(
        probs: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """KL(q || p) + alpha * KL(p_bar || uniform) with soft targets."""
        if probs.shape != target.shape:
            raise ValueError(f"Shape mismatch: probs {probs.shape} vs target {target.shape}")
        # KL(q || p)
        loss_primary = (target * (torch.log(target + eps) - torch.log(probs + eps))).sum(dim=-1).mean()
        if alpha <= 0:
            return loss_primary
        p_bar = probs.mean(dim=0)
        uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
        reg = (p_bar * (torch.log(p_bar + eps) - torch.log(uniform + eps))).sum()
        return loss_primary + alpha * reg

    def fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.1,
    ) -> float:
        """Post-hoc temperature calibration on held-out data."""
        if logits.shape[0] == 0:
            raise ValueError("Need non-empty logits to calibrate temperature")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have the same batch dimension")
        device = logits.device
        dtype = logits.dtype
        log_temp = torch.log(torch.tensor(self._temperature, device=device, dtype=dtype))
        log_temp.requires_grad_()
        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            scaled = logits / temp
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._temperature = float(torch.exp(log_temp).detach().cpu().item())
        return self._temperature

    @staticmethod
    def abstain_mask(confidences: torch.Tensor, threshold: float) -> torch.Tensor:
        """Return boolean mask for samples that should abstain."""
        if threshold <= 0:
            return torch.zeros_like(confidences, dtype=torch.bool)
        return confidences < threshold
