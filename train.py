import argparse
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

from dataset import (
    CloudRemovalDataset,
    RainRemovalDataset,
    SnowRemovalDataset,
    TaskLabeledDataset,
)
from models.DehazeXL.decoders.decoder import DehazeXL


def perceptual_loss(output: torch.Tensor, target: torch.Tensor, vgg: nn.Module) -> torch.Tensor:
    """Perceptual loss using VGG16 features on resized inputs."""
    output = F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)
    target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
    vgg_out = vgg(output)
    vgg_tgt = vgg(target)
    return F.l1_loss(vgg_out, vgg_tgt)


def freq_loss(res, clear, criterion):
    """Compute frequency-domain L1 loss with orthonormal FFT scaling."""

    clear_fft = torch.fft.fft2(clear, dim=(-2, -1), norm="ortho")
    pred_fft = torch.fft.fft2(res, dim=(-2, -1), norm="ortho")
    clear_fft = torch.view_as_real(clear_fft)
    pred_fft = torch.view_as_real(pred_fft)
    loss = criterion(pred_fft, clear_fft)
    return loss


def cal_loss(
    res: torch.Tensor,
    clear_img: torch.Tensor,
    criterion: nn.Module,
    vgg: Optional[nn.Module] = None,
    aux_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    aux_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss1 = criterion(res, clear_img)
    loss2 = freq_loss(res, clear_img, criterion)
    loss = loss1 + loss2
    components = {
        "l1": loss1.detach().item(),
        "freq": loss2.detach().item(),
    }
    if vgg is not None:
        loss3 = perceptual_loss(res, clear_img, vgg)
        loss = loss + 0.2 * loss3  # 可调权重
        components["perceptual"] = loss3.detach().item()
    if aux_info is not None and aux_weight > 0:
        aux_pred, aux_target = aux_info
        aux_loss = F.mse_loss(aux_pred, aux_target)
        loss = loss + aux_weight * aux_loss
        components["aux"] = aux_loss.detach().item()
    components["total"] = loss.detach().item()
    return loss, components


def normalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key in {"8k", "dehaze", "haze", "cloud"}:
        return "dehaze"
    if key in {"rain", "derain"}:
        return "derain"
    if key in {"snow", "desnow"}:
        return "desnow"
    if key == "composite":
        return "composite"
    raise ValueError(f"Unknown dataset name: {name}")


def parse_kv_pairs(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not text:
        return mapping
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid key=value pair: {chunk}")
        key, value = chunk.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def make_single_task_dataset(task: str, root_dir: str, normalize: bool, crop_size: int) -> Dataset:
    task = normalize_dataset_name(task)
    if task == "dehaze":
        base = CloudRemovalDataset(root_dir, normalize=normalize, crop_size=crop_size)
    elif task == "derain":
        base = RainRemovalDataset(root_dir, normalize=normalize, crop_size=crop_size)
    elif task == "desnow":
        base = SnowRemovalDataset(root_dir, normalize=normalize, crop_size=crop_size)
    else:
        raise ValueError(f"Unsupported task for dataset creation: {task}")
    return TaskLabeledDataset(base, task)


def build_dataset(args) -> Tuple[Dataset, list[str]]:
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name == "composite":
        mapping = parse_kv_pairs(args.composite_roots)
        if len(mapping) == 0:
            raise ValueError("Provide --composite_roots task=path pairs for composite dataset")
        datasets = []
        tasks = []
        for task, path in mapping.items():
            normalized_task = normalize_dataset_name(task)
            datasets.append(make_single_task_dataset(normalized_task, path, args.nor, args.crop_size))
            tasks.append(normalized_task)
        concat = ConcatDataset(datasets)
        return concat, sorted(set(tasks))
    dataset = make_single_task_dataset(dataset_name, args.data_dir, args.nor, args.crop_size)
    return dataset, [dataset_name]


def configure_model_for_mode(
    model: DehazeXL,
    mode: str,
    task: Optional[str],
    adapter_reduction: float,
    unfreeze_encoder_stages: int,
) -> None:
    mode = mode.lower()
    if mode == "expert":
        if task is None:
            raise ValueError("Expert mode requires specifying --task")
        task = normalize_dataset_name(task)
        model.register_adapter(task, r=adapter_reduction)
        model.set_active_task(task)
        for param in model.parameters():
            param.requires_grad = False
        # Enable adapters and high-level modules
        for param in model.task_adapters[task].parameters():
            param.requires_grad = True
        for module in (model.bottleneck, model.decoder, model.descriptor, model.global_aux_head):
            for param in module.parameters():
                param.requires_grad = True
        if unfreeze_encoder_stages > 0:
            encoder_inner = getattr(model.encoder, "model", None)
            if encoder_inner is not None:
                for attr in ("stages", "layers"):
                    modules = getattr(encoder_inner, attr, None)
                    if modules is None:
                        continue
                    if isinstance(modules, nn.ModuleList):
                        modules_list = list(modules)
                    elif isinstance(modules, (list, tuple)):
                        modules_list = list(modules)
                    else:
                        modules_list = list(modules.children())
                    if len(modules_list) == 0:
                        continue
                    for stage in modules_list[-unfreeze_encoder_stages:]:
                        for param in stage.parameters():
                            param.requires_grad = True
                    break
    else:
        model.set_active_task(None)
        for param in model.parameters():
            param.requires_grad = True


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    current_state = target_model.state_dict()
    new_state: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module.") and key[7:] in current_state:
            new_state[key[7:]] = value
        elif (not key.startswith("module.")) and ("module." + key) in current_state:
            new_state["module." + key] = value
        elif key in current_state:
            new_state[key] = value
    missing, unexpected = target_model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, scheduler, args, vgg):
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint {args.resume} not found.')
        print(f'Resume from {args.resume}')
        load_checkpoint(model, args.resume)
        evaluate(model, test_dataloader, criterion, device, vgg, args)
    else:
        print('Train from scratch...')
    print("Learning rate: ", args.lr)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    best_loss = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()

        losses: Dict[str, list] = defaultdict(list)
        for batch in tqdm(train_dataloader, desc='Training'):
            step += 1
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)

            if args.aux_weight > 0:
                res, _, aux_pred, aux_target = model(cloud_imgs, return_aux=True)
                aux_info = (aux_pred, aux_target)
            else:
                res = model(cloud_imgs)
                aux_info = None
            loss, comp = cal_loss(
                res,
                clear_imgs,
                criterion,
                vgg,
                aux_info=aux_info,
                aux_weight=args.aux_weight,
            )

            if torch.isnan(loss):
                raise FloatingPointError(
                    "Loss became NaN. Consider lowering the learning rate, "
                    "reducing crop size, or enabling gradient clipping."
                )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            optimizer.step()

            scheduler.step()
            for k, v in comp.items():
                losses[k].append(v)
            if step % int(args.save_cycle * len(train_dataloader)) == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(step)))
        epoch_loss = np.mean(losses.get("total", [0.0]))
        print(f'Epoch:{epoch + 1}/{args.epochs} | Loss:{epoch_loss:.4f}')
        test_loss = evaluate(model, test_dataloader, criterion, device, vgg, args)
        if test_loss <= best_loss:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
            best_loss = test_loss
            print('Saving best model...')

    print('\nTrain Complete.\n')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'last.pth'))


def evaluate(model, test_dataloader, criterion, device, vgg, args):
    model.eval()
    losses: Dict[str, list] = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            if args.aux_weight > 0:
                res, _, aux_pred, aux_target = model(cloud_imgs, return_aux=True)
                aux_info = (aux_pred, aux_target)
            else:
                res = model(cloud_imgs)
                aux_info = None
            loss, comp = cal_loss(
                res,
                clear_imgs,
                criterion,
                vgg,
                aux_info=aux_info,
                aux_weight=args.aux_weight,
            )
            for k, v in comp.items():
                losses[k].append(v)
    mean_loss = np.mean(losses.get("total", [0.0]))
    print(f'Test | Loss:{mean_loss:.4f}')
    return mean_loss


def main(args):
    args.dataset = normalize_dataset_name(args.dataset)
    dataset, dataset_tasks = build_dataset(args)
    if args.mode == "expert":
        task_name = normalize_dataset_name(args.task)
        if task_name not in dataset_tasks:
            raise ValueError(f"Expert task {task_name} not present in dataset tasks {dataset_tasks}")
        args.task = task_name

    generator = torch.Generator().manual_seed(args.seed)
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        train_len = len(dataset) - 1
        val_len = 1
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    base_model = DehazeXL(global_dim=args.global_dim).to(device)
    configure_model_for_mode(
        base_model,
        args.mode,
        getattr(args, "task", None),
        adapter_reduction=args.adapter_reduction,
        unfreeze_encoder_stages=args.unfreeze_encoder_stages,
    )
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")

    model = base_model
    if torch.cuda.device_count() > 1 and not args.no_cuda:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(base_model)

    criterion = nn.L1Loss().to(device)
    if args.disable_perceptual:
        vgg = None
    else:
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )
    optimizer.zero_grad()
    train_model(model, train_loader, val_loader, optimizer, criterion, device, scheduler, args, vgg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, default=r'./datasets/8KDehaze',
                        help='Path to dataset (single-task training)')
    parser.add_argument('--composite_roots', type=str, default='',
                        help='Comma separated task=path pairs for composite training (e.g., dehaze=...,derain=...)')
    parser.add_argument('--save_dir', type=str, default=r'./checkpoints/train',
                        help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=1,
                        help='Cycle of saving checkpoint (fraction of an epoch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    parser.add_argument('--mode', type=str, choices=['dense', 'expert'], default='dense',
                        help='Training mode: dense (shared) or expert (task-specific adapters)')
    parser.add_argument('--task', type=str, default='dehaze',
                        help='Task to train (dehaze/derain/desnow); used when mode=expert')
    parser.add_argument('--dataset', type=str, default='dehaze',
                        help='Dataset preset: dehaze/derain/desnow/composite')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=22,
                        help='Random seed for dataset split')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize the image')
    parser.add_argument('--crop_size', type=int, default=4096,
                        help='Crop size of patches (set 0 to disable)')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Clip gradient norm to this value (0 to disable)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--adapter_reduction', type=float, default=1 / 16,
                        help='Adapter bottleneck reduction ratio r')
    parser.add_argument('--unfreeze_encoder_stages', type=int, default=1,
                        help='Number of top encoder stages to unfreeze for expert fine-tuning')
    parser.add_argument('--aux_weight', type=float, default=1e-3,
                        help='Weight for global descriptor auxiliary loss')
    parser.add_argument('--global_dim', type=int, default=1024,
                        help='Dimensionality of global descriptor vector')
    parser.add_argument('--disable_perceptual', action='store_true', default=False,
                        help='Disable perceptual loss term')
    main(parser.parse_args())
