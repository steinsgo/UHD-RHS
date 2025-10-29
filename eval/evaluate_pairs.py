import argparse
import json
import os
from datetime import datetime
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from models.DehazeXL.decoders.decoder import DehazeXL
from eval.metrics import psnr, ssim
from infer.sliding_window import sliding_window_infer


def list_images(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Build model and adapters
    model = DehazeXL().to(device)
    model.register_adapter('dehaze')
    model.register_adapter('derain')
    model.set_active_task(args.task)

    # Load weights (handle DataParallel prefix)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    state_dict = torch.load(args.model, map_location='cpu')
    new_state = {}
    model_keys = set(model.state_dict().keys())
    for k, v in state_dict.items():
        if k.startswith('module.') and k[7:] in model_keys:
            new_state[k[7:]] = v
        elif (not k.startswith('module.')) and ('module.' + k) in model_keys:
            new_state['module.' + k] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.eval()

    # Pre/Post transforms
    t_list = [transforms.ToTensor()]
    if args.nor:
        # Keep in sync with test.py normalization constants
        t_list.append(transforms.Normalize(
            [0.48694769, 0.53490934, 0.45040282],
            [0.16451647, 0.15308164, 0.15011494]
        ))
    to_tensor = transforms.Compose(t_list)

    def maybe_resize(img: Image.Image) -> Image.Image:
        if args.uni_size_w > 0 and args.uni_size_h > 0:
            return img.resize((args.uni_size_w, args.uni_size_h))
        return img

    names = list_images(args.input_dir)
    if len(names) == 0:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    psnr_vals = []
    ssim_vals = []

    per_image = []
    with torch.no_grad():
        for name in names:
            in_path = os.path.join(args.input_dir, name)
            gt_path = os.path.join(args.gt_dir, name)
            if not os.path.exists(gt_path):
                print(f"[Skip] GT not found for {name}")
                continue
            inp = Image.open(in_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')

            inp = maybe_resize(inp)
            gt = maybe_resize(gt)

            x = to_tensor(inp).unsqueeze(0).to(device)
            y = transforms.ToTensor()(gt).unsqueeze(0).to(device)

            if args.fp16:
                model.half()
                x = x.half()

            # inference: direct or sliding window
            if args.tile > 0:
                def fwd(p):
                    return model(p)
                out = sliding_window_infer(fwd, x, tile=args.tile, overlap=args.overlap, blend=args.blend)
            else:
                out = model(x)

            # If normalized input was used, out is in normalized space; unnormalize for metrics
            if args.nor:
                mean = torch.tensor([0.48694769, 0.53490934, 0.45040282], device=out.device, dtype=out.dtype).view(1, -1, 1, 1)
                std = torch.tensor([0.16451647, 0.15308164, 0.15011494], device=out.device, dtype=out.dtype).view(1, -1, 1, 1)
                out = out * std + mean

            out = out.clamp(0, 1)

            # optional crop borders
            if args.crop > 0:
                c = args.crop
                out = out[:, :, c:-c, c:-c]
                y = y[:, :, c:-c, c:-c]

            # optional Y-channel metrics
            def to_y(t: torch.Tensor) -> torch.Tensor:
                # BT.601 luma approx
                w = torch.tensor([0.299, 0.587, 0.114], device=t.device, dtype=t.dtype).view(1, 3, 1, 1)
                if t.shape[1] == 3:
                    ych = (t * w).sum(dim=1, keepdim=True)
                    return ych
                return t

            pred_m = out.float()
            gt_m = y.float()
            if args.y_only:
                pred_m = to_y(pred_m)
                gt_m = to_y(gt_m)

            p = psnr(pred_m, gt_m).item()
            s = ssim(pred_m, gt_m).item()
            psnr_vals.append(p)
            ssim_vals.append(s)
            per_image.append({"name": name, "psnr": p, "ssim": s})

    if len(psnr_vals) == 0:
        print("No valid pairs evaluated.")
        return
    mean_psnr = sum(psnr_vals)/len(psnr_vals)
    mean_ssim = sum(ssim_vals)/len(ssim_vals)
    print(f"PSNR: {mean_psnr:.4f}")
    print(f"SSIM: {mean_ssim:.4f}")

    # optional JSON report
    if args.report:
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": os.path.abspath(args.model),
            "task": args.task,
            "input_dir": os.path.abspath(args.input_dir),
            "gt_dir": os.path.abspath(args.gt_dir),
            "tile": args.tile,
            "overlap": args.overlap,
            "blend": args.blend,
            "crop": args.crop,
            "y_only": args.y_only,
            "metrics": {"psnr": mean_psnr, "ssim": mean_ssim},
            "per_image": per_image,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM on paired folders")
    parser.add_argument('--input_dir', type=str, required=True, help='Folder of degraded/rain/haze images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Folder of GT/clean images (same filenames)')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--task', type=str, choices=['dehaze', 'derain'], default='dehaze', help='Expert to use')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--fp16', action='store_true', default=False, help='Inference in FP16')
    parser.add_argument('--nor', action='store_true', default=False, help='Apply same normalization as test.py to inputs')
    parser.add_argument('--uni_size_w', type=int, default=4096, help='Unified width for inference (0 to keep)')
    parser.add_argument('--uni_size_h', type=int, default=4096, help='Unified height for inference (0 to keep)')
    parser.add_argument('--y_only', action='store_true', default=False, help='Compute metrics on Y/luma channel')
    parser.add_argument('--crop', type=int, default=0, help='Crop border pixels before metrics (per side)')
    parser.add_argument('--tile', type=int, default=0, help='Sliding window tile size (0 to disable)')
    parser.add_argument('--overlap', type=int, default=64, help='Sliding window overlap in pixels')
    parser.add_argument('--blend', type=str, choices=['hann', 'none'], default='hann', help='Blending window for tiles')
    parser.add_argument('--report', type=str, default='', help='Path to save JSON report (optional)')
    main(parser.parse_args())
