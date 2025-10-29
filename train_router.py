import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.DehazeXL.modules.router import InstanceRouter


@dataclass
class CacheRecord:
    descriptor: torch.Tensor
    errors: torch.Tensor
    oracle_index: int


def load_cache_file(path: str, expert_order: Sequence[str], metric_mode: str) -> List[CacheRecord]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    records: List[CacheRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            descriptor = torch.tensor(data["descriptor"], dtype=torch.float32)
            if "errors" in data:
                raw = data["errors"]
                errors = [raw[name] for name in expert_order]
            elif "metrics" in data:
                raw = data["metrics"]
                if metric_mode == "score":
                    errors = [-raw[name] for name in expert_order]
                else:
                    errors = [raw[name] for name in expert_order]
            else:
                raise KeyError("Cache entry must contain 'errors' or 'metrics'")
            errors_tensor = torch.tensor(errors, dtype=torch.float32)
            oracle_index = int(torch.argmin(errors_tensor).item())
            records.append(CacheRecord(descriptor=descriptor, errors=errors_tensor, oracle_index=oracle_index))
    if len(records) == 0:
        raise ValueError(f"No records loaded from {path}")
    return records


class RouterCacheDataset(Dataset):
    def __init__(self, cache_paths: Iterable[str], expert_order: Sequence[str], metric_mode: str):
        self.records: List[CacheRecord] = []
        for path in cache_paths:
            self.records.extend(load_cache_file(path, expert_order, metric_mode))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        return rec.descriptor, rec.errors, rec.oracle_index


def compute_soft_targets(errors: torch.Tensor, beta: float) -> torch.Tensor:
    scaled = torch.exp(-beta * (errors - errors.min(dim=-1, keepdim=True).values))
    return scaled / scaled.sum(dim=-1, keepdim=True)


def evaluate_router(router: InstanceRouter, loader: DataLoader, beta: float, device: torch.device) -> Dict[str, float]:
    router.eval()
    total_loss = 0.0
    total_agreement = 0.0
    count = 0
    with torch.no_grad():
        for descriptors, errors, oracle_idx in loader:
            descriptors = descriptors.to(device)
            errors = errors.to(device)
            oracle_idx = oracle_idx.to(device)
            target = compute_soft_targets(errors, beta)
            probs, _ = router.forward(descriptors, return_logits=False)
            loss = InstanceRouter.routing_loss(probs, target)
            pred_idx = probs.argmax(dim=-1)
            agreement = (pred_idx == oracle_idx).float().mean().item()
            total_loss += loss.item() * descriptors.size(0)
            total_agreement += agreement * descriptors.size(0)
            count += descriptors.size(0)
    return {
        "loss": total_loss / count,
        "agreement": total_agreement / count,
    }


def collate_fn(batch):
    descriptors, errors, oracle_idx = zip(*batch)
    return torch.stack(descriptors), torch.stack(errors), torch.tensor(oracle_idx, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train instance-level router from cached utilities.")
    parser.add_argument("--train_cache", type=str, nargs="+", required=True, help="One or more JSONL cache files for training")
    parser.add_argument("--val_cache", type=str, nargs="+", default=[], help="Optional cache files for validation/calibration")
    parser.add_argument("--experts", type=str, nargs="+", required=True, help="Ordered list of expert names (e.g., dehaze derain desnow)")
    parser.add_argument("--input_dim", type=int, default=1024, help="Descriptor dimensionality")
    parser.add_argument("--hidden_dim", type=int, default=192, help="Router hidden size")
    parser.add_argument("--beta", type=float, default=50.0, help="Utility distillation temperature (beta)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Load-balance regularizer weight")
    parser.add_argument("--metric_mode", choices=["error", "score"], default="error", help="Indicates whether cached values are errors (lower better) or scores (higher better)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for router training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Router learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--calibrate_temperature", action="store_true", help="Enable temperature scaling using validation cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained router checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    expert_order = args.experts
    train_dataset = RouterCacheDataset(args.train_cache, expert_order, args.metric_mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = None
    if args.val_cache:
        val_dataset = RouterCacheDataset(args.val_cache, expert_order, args.metric_mode)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        )

    router = InstanceRouter(
        input_dim=args.input_dim,
        num_experts=len(expert_order),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        router.train()
        epoch_loss = 0.0
        epoch_agreement = 0.0
        samples = 0
        for descriptors, errors, oracle_idx in tqdm(train_loader, desc=f"Epoch {epoch}"):
            descriptors = descriptors.to(device)
            errors = errors.to(device)
            oracle_idx = oracle_idx.to(device)
            target = compute_soft_targets(errors, args.beta)
            probs, _ = router.forward(descriptors, return_logits=False)
            loss = InstanceRouter.routing_loss(probs, target, alpha=args.alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_idx = probs.argmax(dim=-1)
            batch_size = descriptors.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_agreement += (pred_idx == oracle_idx).float().sum().item()
            samples += batch_size

        avg_loss = epoch_loss / samples
        avg_agreement = epoch_agreement / samples
        log_message = f"[Epoch {epoch}] loss={avg_loss:.4f} agreement={avg_agreement:.4f}"
        if val_loader is not None:
            val_metrics = evaluate_router(router, val_loader, args.beta, device)
            log_message += f" | val_loss={val_metrics['loss']:.4f} val_agree={val_metrics['agreement']:.4f}"
        print(log_message)

    if args.calibrate_temperature and val_loader is not None:
        router.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for descriptors, _, oracle_idx in val_loader:
                descriptors = descriptors.to(device)
                probs, logits = router.forward(descriptors, temperature=1.0, return_logits=True)
                logits_list.append(logits.cpu())
                labels_list.append(oracle_idx)
        logits_tensor = torch.cat(logits_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
        temperature = router.fit_temperature(logits_tensor, labels_tensor)
        print(f"Calibrated router temperature: {temperature:.4f}")

    checkpoint = {
        "state_dict": router.state_dict(),
        "temperature": router.temperature,
        "experts": expert_order,
        "beta": args.beta,
        "alpha": args.alpha,
        "config": vars(args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Router saved to {args.output}")


if __name__ == "__main__":
    main()
