# scripts/eval_adv.py
import os, sys

# ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json, argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.unet import UNet
from utils.dataset import ISICDataset
from utils.metrics import dice_score_numpy, iou_score_numpy
from scripts.attacks.fgsm_seg import fgsm_attack_on_segmentation
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation
from scripts.attacks.bim_seg import bim_attack_on_segmentation
from scripts.attacks.mim_seg import mim_attack_on_segmentation

# defense imports
from utils.defense_tta import (
    median_denoise_tensor,
    gaussian_denoise_tensor,
    bit_depth_reduce_tensor,
    tta_predict,
    _make_identity_transform,
    _make_flip_transform,
    _make_shift_transform
)


def load_model(ckpt_path, device, cfg):
    """
    Load model with auto-detection for 'denoise' mode.
    """
    base_ch = cfg.get('base_ch', 16)
    in_ch = cfg.get('in_channels', 3)
    n_classes = cfg.get('num_classes', 2)

    # 1. Load the state dictionary first to inspect keys
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if 'state_dict' in state:
            sd = state['state_dict']
        elif 'model_state_dict' in state:
            sd = state['model_state_dict']
        else:
            sd = state
    else:
        sd = state

    # 2. Strip "module." prefix if present
    new_sd = {}
    for k, v in sd.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_sd[new_key] = v

    # 3. Auto-detect if the loaded weights contain denoising layers
    # Check for keys like 'dn2.g.weight', 'dn3.g.weight', etc.
    has_denoise_layers = any(k.startswith('dn2.') or k.startswith('dn3.') for k in new_sd.keys())

    if has_denoise_layers:
        print(
            f"[INFO] Auto-detected Feature Denoising layers in checkpoint '{os.path.basename(ckpt_path)}'. Enabling denoise=True.")
        use_denoise = True
    else:
        # Fallback to config or default False
        use_denoise = cfg.get('denoise', False)
        if use_denoise:
            print(f"[INFO] Config requests denoise=True, but checkpoint might not have them? (Check strict mode below)")

    # 4. Initialize Model with correct denoise setting
    model = UNet(
        in_channels=in_ch,
        n_classes=n_classes,
        base_ch=base_ch,
        denoise=use_denoise
    ).to(device)

    # 5. Load weights
    try:
        model.load_state_dict(new_sd, strict=True)
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed: {e}")
        print("[WARN] Retrying with strict=False (some keys might be missing or unexpected)")
        model.load_state_dict(new_sd, strict=False)

    model.eval()
    return model


def make_file_list_from_dir(images_dir):
    import glob
    patterns = ['ISIC_*.jpg', 'ISIC_*.jpeg', 'ISIC_*.png']
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(images_dir, p)))
    all_files = sorted(list(set(all_files)))
    all_files = [os.path.basename(x) for x in all_files]
    return all_files


def build_val_list(cfg, args):
    use_splits = cfg.get('use_splits', False)
    if use_splits:
        with open('splits.json') as f:
            splits = json.load(f)
        val_list = splits.get('val', [])
        return val_list

    images_dir = cfg['images_dir']
    all_files = make_file_list_from_dir(images_dir)
    if len(all_files) == 0:
        return []

    if args.val_start_name:
        name = args.val_start_name
        # normalize check
        if name not in all_files:
            # try finding by base name
            base = os.path.splitext(name)[0]
            matches = [f for f in all_files if os.path.splitext(f)[0] == base]
            if matches:
                start_idx = all_files.index(matches[0])
            else:
                raise FileNotFoundError(f"val_start_name {name} not found in {images_dir}")
        else:
            start_idx = all_files.index(name)
    else:
        start_idx = args.val_start if args.val_start is not None else cfg.get('val_start', 0)
        if start_idx < 0: start_idx = 0

    val_count = args.val_count if args.val_count is not None else cfg.get('val_count', 20)
    val_list = all_files[start_idx: start_idx + val_count]
    return val_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--attack', choices=['none', 'fgsm', 'pgd', 'bim', 'mim'], default='none')
    parser.add_argument('--eps', type=float, default=4 / 255)

    # Attack params
    parser.add_argument('--attack_iters', type=int, default=None)
    parser.add_argument('--attack_alpha', type=float, default=None)
    parser.add_argument('--attack_decay', type=float, default=None)

    # Data params
    parser.add_argument('--val_start', type=int, default=None)
    parser.add_argument('--val_start_name', type=str, default=None)
    parser.add_argument('--val_count', type=int, default=None)

    # Defense params (Inference time)
    parser.add_argument('--defense', choices=['none', 'denoise', 'tta'], default='none',
                        help='Enable inference-time defense (input preprocessing)')
    parser.add_argument('--defense_bits', type=int, default=6)
    parser.add_argument('--tta_transforms', type=int, default=3)

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')

    # Build validation list
    try:
        val_list = build_val_list(cfg, args)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not val_list:
        print("[WARN] No validation files found. Creating empty CSV and exiting.")
        out_csv = os.path.join(cfg.get('out_dir', 'outputs'), f'eval_{args.attack}_eps{args.eps:.4f}.csv')
        os.makedirs(cfg.get('out_dir', 'outputs'), exist_ok=True)
        pd.DataFrame(columns=['dice', 'iou']).to_csv(out_csv, index=False)
        sys.exit(0)

    print(f"Evaluating {len(val_list)} samples starting from {val_list[0]}")

    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Load Model (Auto-detects denoise layers)
    model = load_model(args.ckpt, device, cfg)

    # Attack configuration
    iters = args.attack_iters if args.attack_iters is not None else cfg.get('attack_iters', None)
    if iters is None:
        iters = cfg.get('attack_iters', 10) if args.attack in ('bim', 'mim', 'pgd') else 1

    alpha = args.attack_alpha if args.attack_alpha is not None else cfg.get('attack_alpha', None)
    if alpha is None:
        alpha = args.eps / float(iters) if iters > 0 else args.eps

    decay = args.attack_decay if args.attack_decay is not None else cfg.get('attack_decay', 1.0)

    print(f"Attack: {args.attack} | eps: {args.eps:.4f} | iters: {iters} | alpha: {alpha:.4f}")

    # Defense configuration (Input Preprocessing / TTA)
    defense_mode = args.defense


    def prepare_transforms(n_transforms):
        transforms = [_make_identity_transform()]
        if n_transforms >= 2: transforms.append(_make_flip_transform('h'))
        if n_transforms >= 3: transforms.append(_make_shift_transform(2, 0))
        return transforms


    if defense_mode != 'none':
        denoise_funcs = [
            lambda x: median_denoise_tensor(x, kernel_size=3),
            lambda x: bit_depth_reduce_tensor(x, bits=args.defense_bits)
        ]
        tta_transforms = prepare_transforms(args.tta_transforms)
    else:
        denoise_funcs = None
        tta_transforms = None

    results = []
    loss_ce = torch.nn.CrossEntropyLoss()

    for img, mask in tqdm(val_loader, total=len(val_loader)):
        img_orig = img.clone()
        mask_np = mask.numpy()[0]
        mask_dev = mask.to(device)

        # 1. Generate Adversarial Example (img_in)
        if args.attack == 'none':
            img_in = img_orig
        elif args.attack == 'fgsm':
            img_in = fgsm_attack_on_segmentation(model, img_orig, mask, args.eps, lambda x, y: loss_ce(x, y.to(device)),
                                                 device)
        elif args.attack == 'pgd':
            img_in = pgd_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters,
                                                lambda x, y: loss_ce(x, y.to(device)), device)
        elif args.attack == 'bim':
            img_in = bim_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters,
                                                lambda x, y: loss_ce(x, y.to(device)), device)
        elif args.attack == 'mim':
            img_in = mim_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters,
                                                lambda x, y: loss_ce(x, y.to(device)), device, decay=decay)
        else:
            raise ValueError(f"Unknown attack: {args.attack}")

        # 2. Inference (with optional input defense)
        if defense_mode == 'none':
            with torch.no_grad():
                logits = model(img_in.to(device))
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
        else:
            # TTA or Denoise inference
            agg_pred, cons = tta_predict(model, img_in.cpu(), device, tta_transforms, denoise_funcs=denoise_funcs,
                                         aggregate='vote')
            pred = agg_pred.cpu().numpy()

        dice = dice_score_numpy(pred, mask_np)
        iou = iou_score_numpy(pred, mask_np)
        results.append({'dice': float(dice), 'iou': float(iou)})

    df = pd.DataFrame(results)
    if df.empty:
        print("No results computed.")
    else:
        print("\nResults Summary:")
        print(df.describe())

    # Generate output filename
    out_name = f'eval_{args.attack}'
    if args.attack != 'none':
        out_name += f'_eps{args.eps:.4f}'
    if defense_mode != 'none':
        out_name += f'_def{defense_mode}'
    out_name += '.csv'

    out_csv = os.path.join(cfg.get('out_dir', 'outputs'), out_name)
    os.makedirs(cfg.get('out_dir', 'outputs'), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")