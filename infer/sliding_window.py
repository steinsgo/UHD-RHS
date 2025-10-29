import torch
import torch.nn.functional as F


def _hann_1d(length: int, device=None, dtype=None):
    if length <= 1:
        return torch.ones(length, device=device, dtype=dtype)
    n = torch.arange(length, device=device, dtype=dtype)
    return 0.5 * (1 - torch.cos(2 * torch.pi * n / (length - 1)))


def _hann_2d(h: int, w: int, device=None, dtype=None):
    hy = _hann_1d(h, device=device, dtype=dtype)
    hx = _hann_1d(w, device=device, dtype=dtype)
    win = torch.ger(hy, hx)  # [h, w]
    # normalize peak to 1
    if win.max() > 0:
        win = win / win.max()
    return win


@torch.no_grad()
def sliding_window_infer(
    model,
    x: torch.Tensor,
    tile: int = 512,
    overlap: int = 64,
    blend: str = "hann",
):
    """Run sliding-window inference with overlap and optional Hann blending.

    Args:
        model: callable mapping NCHW -> NCHW
        x: input tensor [N, C, H, W] (supports N=1 best)
        tile: window size (clamped to min(H, W))
        overlap: pixels of overlap between tiles
        blend: 'hann' or 'none'
    Returns:
        y: output tensor [N, C, H, W]
    """
    assert x.dim() == 4, "x should be NCHW"
    n, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    tile_h = min(tile, h)
    tile_w = min(tile, w)
    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    if blend == "hann":
        win2d = _hann_2d(tile_h, tile_w, device=device, dtype=dtype)
        win2d = win2d.unsqueeze(0).unsqueeze(0)  # [1,1,th,tw]
    else:
        win2d = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=dtype)

    y = torch.zeros(n, c, h, w, device=device, dtype=dtype)
    wsum = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

    ys = list(range(0, h, stride_h))
    xs = list(range(0, w, stride_w))

    for y0 in ys:
        y0 = min(y0, h - tile_h)
        for x0 in xs:
            x0 = min(x0, w - tile_w)
            patch = x[:, :, y0:y0 + tile_h, x0:x0 + tile_w]
            out = model(patch)
            y[:, :, y0:y0 + tile_h, x0:x0 + tile_w] += out * win2d
            wsum[:, :, y0:y0 + tile_h, x0:x0 + tile_w] += win2d

    y = y / (wsum.clamp_min(1e-8))
    return y

