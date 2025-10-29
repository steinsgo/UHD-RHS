import argparse
import os
import time

import torch
from PIL import Image
from torchvision import transforms

from models.DehazeXL.decoders.decoder import DehazeXL
from infer.sliding_window import sliding_window_infer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    model = DehazeXL().to(device)
    model.register_adapter('dehaze')
    model.register_adapter('derain')
    model.set_active_task(args.task)

    # Load weights
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
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # Input(s)
    paths = []
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    paths = [p for p in paths if os.path.isfile(p)]
    if len(paths) == 0:
        raise FileNotFoundError("No input images found")

    t_list = [transforms.ToTensor()]
    if args.nor:
        t_list.append(transforms.Normalize(
            [0.48694769, 0.53490934, 0.45040282],
            [0.16451647, 0.15308164, 0.15011494]
        ))
    to_tensor = transforms.Compose(t_list)

    # Warmup
    with torch.no_grad():
        img = Image.open(paths[0]).convert('RGB')
        if args.uni_size_w > 0 and args.uni_size_h > 0:
            img = img.resize((args.uni_size_w, args.uni_size_h))
        x = to_tensor(img).unsqueeze(0).to(device)
        if args.fp16:
            model.half(); x = x.half()
        if args.tile > 0:
            _ = sliding_window_infer(model, x, tile=args.tile, overlap=args.overlap, blend=args.blend)
        else:
            _ = model(x)

    # Measure
    times = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    start_mem = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    with torch.no_grad():
        for p in paths[:args.limit]:
            img = Image.open(p).convert('RGB')
            if args.uni_size_w > 0 and args.uni_size_h > 0:
                img = img.resize((args.uni_size_w, args.uni_size_h))
            x = to_tensor(img).unsqueeze(0).to(device)
            if args.fp16:
                x = x.half()
            t0 = time.time()
            if args.tile > 0:
                _ = sliding_window_infer(model, x, tile=args.tile, overlap=args.overlap, blend=args.blend)
            else:
                _ = model(x)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            times.append(t1 - t0)

    end_mem = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    avg_time = sum(times) / len(times)
    w = args.uni_size_w if args.uni_size_w > 0 else img.width
    h = args.uni_size_h if args.uni_size_h > 0 else img.height
    mp = (w * h) / 1e6
    ms_per_mp = (avg_time * 1000) / mp
    print(f"Images: {len(times)}  Avg Time: {avg_time:.4f}s  ms/MP: {ms_per_mp:.2f}")
    if device.type == 'cuda':
        print(f"Max CUDA Mem: {end_mem/1e6:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile latency/memory/throughput')
    parser.add_argument('--input', type=str, required=True, help='Image path or folder')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--task', type=str, choices=['dehaze', 'derain'], default='dehaze')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--nor', action='store_true', default=False)
    parser.add_argument('--uni_size_w', type=int, default=4096)
    parser.add_argument('--uni_size_h', type=int, default=4096)
    parser.add_argument('--tile', type=int, default=0)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--blend', type=str, choices=['hann', 'none'], default='hann')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of images to profile')
    main(parser.parse_args())

