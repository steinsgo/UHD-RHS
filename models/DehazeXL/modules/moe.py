from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, TYPE_CHECKING

import torch
from torch import nn

from .router import InstanceRouter, RoutingDecision

if TYPE_CHECKING:
    from models.DehazeXL.decoders.decoder import DehazeXL


@dataclass
class MoEForwardMetadata:
    routing: RoutingDecision
    abstain_mask: torch.Tensor
    used_tasks: List[str]
    descriptor: torch.Tensor


class InstanceMoERestore(nn.Module):
    """Wrap DehazeXL with instance-level routing and optional fallback."""

    def __init__(
        self,
        backbone: "DehazeXL",
        router: InstanceRouter,
        expert_tasks: Iterable[str],
        fallback_task: Optional[str] = None,
        abstain_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.router = router
        self.expert_tasks: List[str] = list(expert_tasks)
        if len(self.expert_tasks) != router.num_experts:
            raise ValueError(
                f"Router expects {router.num_experts} experts, got {len(self.expert_tasks)} task names"
            )
        self.fallback_task = fallback_task
        self.abstain_threshold = max(abstain_threshold, 0.0)

    def set_abstain_threshold(self, threshold: float) -> None:
        self.abstain_threshold = max(threshold, 0.0)

    def set_fallback_task(self, task: Optional[str]) -> None:
        self.fallback_task = task

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
        threshold: Optional[float] = None,
        return_metadata: bool = False,
    ):
        features, skip, n_regions = self.backbone.encode(x)
        descriptor = self.backbone.global_descriptor(features)
        routing = self.router.route(descriptor, temperature=temperature)
        thr = self.abstain_threshold if threshold is None else max(threshold, 0.0)
        abstain_mask = self.router.abstain_mask(routing.confidences, thr)

        outputs = torch.zeros_like(skip)
        used_tasks: List[str] = []
        filled = torch.zeros(skip.shape[0], dtype=torch.bool, device=skip.device)

        # Decode per expert group
        for expert_idx, task_name in enumerate(self.expert_tasks):
            mask = (routing.expert_ids == expert_idx) & (~abstain_mask)
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            grouped_features = [feat.index_select(0, indices) for feat in features]
            grouped_skip = skip.index_select(0, indices)
            decoded = self.backbone.decode_from_features(grouped_features, grouped_skip, n_regions, task=task_name)
            outputs.index_copy_(0, indices, decoded)
            used_tasks.append(task_name)
            filled.index_fill_(0, indices, True)

        # Fallback for abstentions (dense unified head or specified task)
        if abstain_mask.any():
            indices = abstain_mask.nonzero(as_tuple=False).squeeze(1)
            grouped_features = [feat.index_select(0, indices) for feat in features]
            grouped_skip = skip.index_select(0, indices)
            decoded = self.backbone.decode_from_features(
                grouped_features,
                grouped_skip,
                n_regions,
                task=self.fallback_task,
            )
            outputs.index_copy_(0, indices, decoded)
            used_tasks.append(self.fallback_task or "dense")
            filled.index_fill_(0, indices, True)

        # Any remaining samples default to fallback task
        missing_mask = ~filled
        if missing_mask.any():
            indices = missing_mask.nonzero(as_tuple=False).squeeze(1)
            grouped_features = [feat.index_select(0, indices) for feat in features]
            grouped_skip = skip.index_select(0, indices)
            decoded = self.backbone.decode_from_features(
                grouped_features,
                grouped_skip,
                n_regions,
                task=self.fallback_task,
            )
            outputs.index_copy_(0, indices, decoded)
            filled.index_fill_(0, indices, True)

        if return_metadata:
            meta = MoEForwardMetadata(
                routing=routing,
                abstain_mask=abstain_mask,
                used_tasks=used_tasks,
                descriptor=descriptor,
            )
            return outputs, meta
        return outputs
