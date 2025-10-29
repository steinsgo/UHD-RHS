import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset import DEFAULT_MEAN, DEFAULT_STD
from eval.metrics import psnr, ssim
from models.DehazeXL.decoders.decoder import DehazeXL
from models.DehazeXL.modules.router import InstanceRouter


try:
    import lpips  # type: ignore
except ImportError:  # pragma: no cover
    lpips = None


def parse_kv_pairs(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not text:
        return mapping
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid key=value pair: {chunk}")
        key, value = chunk.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def normalize_task(name: str) -> str:
    key = name.lower()
    if key in {"dehaze", "haze", "cloud"}:
        return "dehaze"
    if key in {"derain", "rain"}:
        return "derain"
    if key in {"desnow", "snow"}:
        return "desnow"
    raise ValueError(f"Unsupported task name: {name}")


def list_images(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    names = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    return sorted(names)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    current = target.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k in current:
            filtered[k] = v
        elif k.startswith("module.") and k[7:] in current:
            filtered[k[7:]] = v
    missing, unexpected = target.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")


def load_adapter_weights(model: DehazeXL, task: str, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    prefix = f"task_adapters.{task}"
    filtered = {k: v for k, v in state.items() if k.startswith(prefix)}
    if len(filtered) == 0:
        print(f"[warn] No adapter weights for task {task} found in {checkpoint_path}")
        return
    model.load_state_dict(filtered, strict=False)


def unnormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_t + mean_t


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model=None,
) -> Dict[str, float]:
    metrics = {
        "psnr": psnr(pred, target).item(),
        "ssim": ssim(pred, target).item(),
    }
    if lpips_model is not None:
        pred_lp = (pred * 2 - 1).to(lpips_model.device)
        tgt_lp = (target * 2 - 1).to(lpips_model.device)
        metrics["lpips"] = lpips_model(pred_lp, tgt_lp).item()
    return metrics


def expected_calibration_error(confidences: np.ndarray, matches: np.ndarray, bins: int = 15) -> float:
    if confidences.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = confidences.size
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = matches[mask].mean()
        ece += abs(bin_acc - bin_conf) * (count / total)
    return float(ece)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense vs Experts vs MoE routing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder of degraded images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder of ground truth images (matching names)")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Dense/unified checkpoint path")
    parser.add_argument("--expert_checkpoints", type=str, required=True,
                        help="Comma separated task=path mapping for experts (e.g., dehaze=...,derain=...)")
    parser.add_argument("--router_checkpoint", type=str, required=True, help="Router checkpoint path")
    parser.add_argument("--abstain_threshold", type=float, default=0.2, help="Confidence threshold for abstention")
    parser.add_argument("--temperature_override", type=float, default=0.0, help="Optional manual router temperature")
    parser.add_argument("--normalize", action="store_true", help="Apply dataset normalization to inputs")
    parser.add_argument("--crop", type=int, default=0, help="Crop border before metrics")
    parser.add_argument("--global_dim", type=int, default=1024, help="Descriptor dimensionality")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (requires lpips package)")
    parser.add_argument("--ece_bins", type=int, default=15, help="Bins for calibration error")
    parser.add_argument("--report", type=str, default="", help="Optional JSON report path")
    args = parser.parse_args()

    device = torch.device(args.device)
    expert_map_raw = parse_kv_pairs(args.expert_checkpoints)
    if len(expert_map_raw) == 0:
        raise ValueError("Provide at least one expert checkpoint mapping")
    tasks = []
    expert_paths = {}
    for task_name, path in expert_map_raw.items():
        norm_task = normalize_task(task_name)
        tasks.append(norm_task)
        expert_paths[norm_task] = path
    tasks = sorted(set(tasks))

    model = DehazeXL(global_dim=args.global_dim).to(device)
    for task in tasks:
        model.register_adapter(task)
    load_checkpoint(model, args.base_checkpoint)
    for task, path in expert_paths.items():
        load_adapter_weights(model, task, path)
    model.eval()

    router_state = torch.load(args.router_checkpoint, map_location="cpu")
    router = InstanceRouter(input_dim=args.global_dim, num_experts=len(tasks))
    router.load_state_dict(router_state["state_dict"])
    if args.temperature_override > 0:
        router.set_temperature(args.temperature_override)
    elif "temperature" in router_state:
        router.set_temperature(router_state["temperature"])
    router = router.to(device)
    lpips_model = None
    if args.lpips:
        if lpips is None:
            raise ImportError("lpips package is required for LPIPS computation")
        lpips_model = lpips.LPIPS(net='vgg').to(device)
        lpips_model.eval()

    transform_list = [transforms.ToTensor()]
    if args.normalize:
        transform_list.append(transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    to_tensor = transforms.Compose(transform_list)
    to_tensor_gt = transforms.ToTensor()

    mean = DEFAULT_MEAN
    std = DEFAULT_STD

    per_image = []
    actual_psnr = []
    dense_psnr = []
    oracle_psnr = []
    confidences = []
    match_flags = []
    abstain_flags = []

    image_names = list_images(args.input_dir)
    if len(image_names) == 0:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    for name in image_names:
        degraded_path = os.path.join(args.input_dir, name)
        gt_path = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_path):
            print(f"[skip] Missing GT for {name}")
            continue
        degraded = Image.open(degraded_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        x = to_tensor(degraded).unsqueeze(0).to(device)
        y = to_tensor_gt(gt).unsqueeze(0).to(device)

        with torch.no_grad():
            features, skip, n_regions = model.encode(x)
            dense_out = model.decode_from_features(features, skip, n_regions, task=None)
            expert_outputs: Dict[str, torch.Tensor] = {}
            for task in tasks:
                expert_outputs[task] = model.decode_from_features(features, skip, n_regions, task=task)
            descriptor = model.get_global_descriptor(features)
            routing = router.route(descriptor)
            conf = routing.confidences[0].item()
            selected_idx = int(routing.expert_ids[0].item())
            selected_task = tasks[selected_idx]
            abstain = conf < args.abstain_threshold
            if abstain:
                moe_out = dense_out
                used_task = "dense"
            else:
                moe_out = expert_outputs[selected_task]
                used_task = selected_task

        def prepare(t: torch.Tensor) -> torch.Tensor:
            out = t
            if args.normalize:
                out = unnormalize_tensor(out, mean, std)
            if args.crop > 0:
                c = args.crop
                out = out[:, :, c:-c, c:-c]
            return out.clamp(0, 1).cpu()

        pred_dense = prepare(dense_out)
        pred_actual = prepare(moe_out)
        preds_expert = {task: prepare(expert_outputs[task]) for task in tasks}
        target = prepare(y)

        dense_metrics = compute_metrics(pred_dense, target, lpips_model)
        actual_metrics = compute_metrics(pred_actual, target, lpips_model)
        expert_metrics = {task: compute_metrics(preds_expert[task], target, lpips_model) for task in tasks}

        oracle_task = max(expert_metrics.items(), key=lambda kv: kv[1]["psnr"])[0]
        oracle_metrics = expert_metrics[oracle_task]

        per_image.append({
            "name": name,
            "dense": dense_metrics,
            "moe": {
                "metrics": actual_metrics,
                "task": used_task,
                "confidence": conf,
                "abstained": abstain,
            },
            "oracle": {"metrics": oracle_metrics, "task": oracle_task},
            "experts": expert_metrics,
        })

        dense_psnr.append(dense_metrics["psnr"])
        actual_psnr.append(actual_metrics["psnr"])
        oracle_psnr.append(oracle_metrics["psnr"])
        confidences.append(conf)
        abstain_flags.append(abstain)
        match_flags.append(0 if abstain else int(used_task == oracle_task))

    if len(per_image) == 0:
        print("No valid image pairs evaluated.")
        return

    num_images = len(per_image)
    dense_mean = float(np.mean(dense_psnr))
    actual_mean = float(np.mean(actual_psnr))
    oracle_mean = float(np.mean(oracle_psnr))
    oracle_gap = float(np.mean(np.array(oracle_psnr) - np.array(actual_psnr)))
    abstain_rate = float(np.mean(abstain_flags))
    non_abstained = max(1, num_images - sum(abstain_flags))
    agreement = float(sum(1 for flag, match in zip(abstain_flags, match_flags) if not flag and match) / non_abstained)

    confidences_arr = np.array([c for c, flag in zip(confidences, abstain_flags) if not flag])
    matches_arr = np.array([m for m, flag in zip(match_flags, abstain_flags) if not flag], dtype=float)
    ece = expected_calibration_error(confidences_arr, matches_arr, bins=args.ece_bins) if confidences_arr.size > 0 else 0.0

    sorted_indices = np.argsort(-np.array(confidences))
    coverage_curve = []
    for frac in np.linspace(0.1, 1.0, 10):
        k = max(1, int(frac * len(sorted_indices)))
        top_idx = sorted_indices[:k]
        kept = [match_flags[i] for i in top_idx if not abstain_flags[i]]
        acc = float(np.mean(kept)) if len(kept) > 0 else 0.0
        coverage_curve.append({"coverage": float(frac), "accuracy": acc})

    summary = {
        "num_images": num_images,
        "dense_psnr": dense_mean,
        "moe_psnr": actual_mean,
        "oracle_psnr": oracle_mean,
        "oracle_gap": oracle_gap,
        "abstain_rate": abstain_rate,
        "agreement": agreement,
        "ece": ece,
        "coverage_curve": coverage_curve,
    }

    if args.report:
        report = {
            "summary": summary,
            "per_image": per_image,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
