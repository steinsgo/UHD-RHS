import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F
from math import log10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CloudRemovalDataset, RainRemovalDataset
from models.DehazeXL.decoders.decoder import DehazeXL

def perceptual_loss(output, target, vgg):
    # 归一化到 [0,1]，适配VGG输入
    output = torch.nn.functional.interpolate(output, size=(224,224), mode='bilinear', align_corners=False)
    target = torch.nn.functional.interpolate(target, size=(224,224), mode='bilinear', align_corners=False)
    vgg_out = vgg(output)
    vgg_tgt = vgg(target)
    return torch.nn.functional.l1_loss(vgg_out, vgg_tgt)

def calculate_psnr(output, target, max_val=1.0):
    # 假设图像在 [0,1] 区间，如果你是 [0,255] 记得把 max_val 改成 255.0
    mse = F.mse_loss(output, target, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

def _gaussian_window(window_size=11, sigma=1.5, device='cpu', channel=3):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)  # [W,1]
    window_2d = window_1d @ window_1d.t()  # [W,W]
    window_2d = window_2d / window_2d.sum()
    window = window_2d.unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(output, target, window_size=11, sigma=1.5):
    """
    简化版 SSIM：对 3 通道图像计算后再取平均。
    假设 output/target 形状为 [B, C, H, W]，数值在 [0,1]。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    B, C, H, W = output.shape
    device = output.device

    window = _gaussian_window(window_size, sigma, device=device, channel=C)

    mu1 = F.conv2d(output, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(output * output, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(output * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # 按通道和空间平均，然后转成 python float
    return ssim_map.mean().item()

def log_to_file(log_path, msg):
    print(msg)
    try:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write log: {e}")

def freq_loss(res, clear, criterion):
    """Compute frequency-domain L1 loss with orthonormal FFT scaling."""

    clear_fft = torch.fft.fft2(clear, dim=(-2, -1), norm="ortho")
    pred_fft = torch.fft.fft2(res, dim=(-2, -1), norm="ortho")
    clear_fft = torch.view_as_real(clear_fft)
    pred_fft = torch.view_as_real(pred_fft)
    loss = criterion(pred_fft, clear_fft)
    return loss


def cal_loss(res, clear_img, criterion, vgg=None):
    loss1 = criterion(res, clear_img)
    loss2 = freq_loss(res, clear_img, criterion)
    loss = loss1 + loss2
    if vgg is not None:
        loss3 = perceptual_loss(res, clear_img, vgg)
        loss = loss + 0.2 * loss3  # 权重可调
    return loss


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, scheduler, args, vgg):
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint {args.resume} not found.')
        else:
            print(f'Resume from {args.resume}')
            model.load_state_dict(torch.load(args.resume, weights_only=True))
            evaluate(model, test_dataloader, criterion, device)
    else:
        print('Train from scratch...')
    print("Learning rate: ", args.lr)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    best_loss = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        n_train_samples = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)

            res = model(cloud_imgs)
            loss = cal_loss(res, clear_imgs, criterion, vgg)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            bs = cloud_imgs.size(0)
            total_train_loss += loss.item() * bs
            n_train_samples += bs

        avg_train_loss = total_train_loss / n_train_samples
        log_to_file(log_path, f"Train | Epoch:{epoch + 1} | Loss:{avg_train_loss:.4f}")

    # 验证 + 记录
        test_loss, test_psnr, test_ssim = evaluate(
        model, test_dataloader, criterion, device, vgg, log_path=log_path, epoch=epoch + 1
        )

        if test_loss < best_loss:
            best_loss = test_loss
            log_to_file(log_path, f"New best at Epoch {epoch + 1} with Loss:{best_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))

    print('\nTrain Complete.\n')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'last.pth'))


def evaluate(model, test_dataloader, criterion, device, vgg, log_path=None, epoch=None):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)

            outputs = model(cloud_imgs)

            loss = cal_loss(outputs, clear_imgs, criterion, vgg)
            batch_size = cloud_imgs.size(0)

            total_loss += loss.item() * batch_size
            total_psnr += calculate_psnr(outputs, clear_imgs, max_val=1.0) * batch_size
            total_ssim += calculate_ssim(outputs, clear_imgs) * batch_size
            n_samples += batch_size

    avg_loss = total_loss / n_samples
    avg_psnr = total_psnr / n_samples
    avg_ssim = total_ssim / n_samples

    msg = f"Test | Epoch:{epoch if epoch is not None else -1} | " \
          f"Loss:{avg_loss:.4f} | PSNR:{avg_psnr:.2f} dB | SSIM:{avg_ssim:.4f}"
    if log_path is not None:
        log_to_file(log_path, msg)
    else:
        print(msg)

    return avg_loss, avg_psnr, avg_ssim


def main(args):
    if args.dataset == 'rain':
        total_dataset = RainRemovalDataset(os.path.join(args.data_dir), args.nor, crop_size=args.crop_size)
    else:
        total_dataset = CloudRemovalDataset(os.path.join(args.data_dir), args.nor, crop_size=args.crop_size)

    generator = torch.Generator().manual_seed(22)
    train_set, test_set = random_split(total_dataset,
                                       [int(len(total_dataset) * 0.85),
                                        len(total_dataset) - int(len(total_dataset) * 0.85)],
                                      generator=generator)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    model = DehazeXL().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    criterion = nn.L1Loss().to(device)
    # 初始化VGG16感知网络
    vgg = vgg16(weights=True).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(params=filter(lambda x: x.requires_grad, model.parameters()),
                                  lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                           eta_min=1e-6)
    optimizer.zero_grad()
    train_model(model, train_loader, test_loader, optimizer, criterion, device, scheduler, args, vgg)

    log_path = os.path.join(args.save_dir, "log.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    log_to_file(log_path, f"Start training with args: {args}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, default=r'./datasets/8KDehaze',
                        help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./checkpoints/train',
                        help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=1,
                        help='Cycle of saving checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Epochs')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize the image')
    parser.add_argument('--dataset', type=str, choices=['8k', 'rain'], default='8k',
                        help='Dataset type to use')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop size of patches (set 0 to disable)')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Clip gradient norm to this value (0 to disable)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    main(parser.parse_args())
