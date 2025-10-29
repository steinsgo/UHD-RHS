import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR per-batch (returns scalar mean).

    pred, target: shape [N, C, H, W], values in [0, data_range].
    """
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    # Avoid log(0)
    eps = 1e-10
    psnr_vals = 10 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr_vals.mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)  # [1, W]
    window_2d = (g.T @ g).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    return window_2d


@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute SSIM per-batch averaged over channels (returns scalar mean).

    pred, target: [N, C, H, W] in [0, data_range].
    Implementation follows standard Gaussian SSIM with constants per data_range.
    """
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    device = pred.device
    dtype = pred.dtype
    window = _gaussian_window(window_size, sigma, device, dtype)
    padding = window_size // 2

    # Per-channel conv by grouping
    N, C, H, W = pred.shape
    window = window.expand(C, 1, window_size, window_size)

    mu_x = F.conv2d(pred, window, padding=padding, groups=C)
    mu_y = F.conv2d(target, window, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    # Average over channels and spatial
    ssim_per_img = ssim_map.mean(dim=(1, 2, 3))
    return ssim_per_img.mean()

