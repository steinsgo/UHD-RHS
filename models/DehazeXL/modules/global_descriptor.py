import torch
from torch import nn


class MultiScaleGlobalDescriptor(nn.Module):
    """Aggregate multi-resolution encoder features into a compact global vector.

    The descriptor pools each encoder feature map to a token, concatenates them,
    and refines the representation with a lightweight MLP followed by LayerNorm.
    """

    def __init__(
        self,
        stage_channels: list[int],
        hidden_dim: int = 1024,
        mlp_hidden_multiplier: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(stage_channels) == 0:
            raise ValueError("stage_channels must be non-empty")
        self.stage_channels = stage_channels
        self.hidden_dim = hidden_dim
        in_dim = sum(stage_channels)
        mlp_dim = int(hidden_dim * mlp_hidden_multiplier)

        self.norm_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim, bias=True),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """features: list of tensors [N, C_i, H_i, W_i] aligned with stage_channels"""
        if len(features) != len(self.stage_channels):
            raise ValueError(
                f"Expected {len(self.stage_channels)} features, got {len(features)}"
            )
        pooled = []
        for feat, expected_c in zip(features, self.stage_channels):
            if feat.shape[1] != expected_c:
                raise ValueError(
                    f"Feature channel mismatch: expected {expected_c}, got {feat.shape[1]}"
                )
            token = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            pooled.append(token)
        concat = torch.cat(pooled, dim=1)
        concat = self.norm_in(concat)
        global_vec = self.mlp(concat)
        global_vec = self.norm_out(global_vec)
        return global_vec

