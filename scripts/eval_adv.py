# scripts/eval_adv.py
# Robust evaluation script: supports val_start index or val_start_name, and val_count.
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
    base_ch = cfg.get('base_ch', 16)
    in_ch = cfg.get('in_channels', 3)
    n_classes = cfg.get('num_classes', 2)
    model = UNet(in_channels=in_ch, n_classes=n_classes, base_ch=base_ch).to(device)

    # robust loading
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

    # strip "module." if present
    new_sd = {}
    for k, v in sd.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_sd[new_key] = v
    model.load_state_dict(new_sd)
    model.eval()
    return model

def make_file_list_from_dir(images_dir):
    import glob
    # 修改为读取所有常见的图片格式，不再限制文件名必须包含 ISIC
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(images_dir, p)))
    all_files = sorted(list(set(all_files)))
    all_files = [os.path.basename(x) for x in all_files]
    return all_files

def build_val_list(cfg, args):
    use_splits = cfg.get('use_splits', False)
    if use_splits:
        # if user insists on splits.json
        with open('splits.json') as f:
            splits = json.load(f)
        val_list = splits.get('val', [])
        return val_list

    # else construct from images_dir
    images_dir = cfg['images_dir']
    all_files = make_file_list_from_dir(images_dir)
    if len(all_files) == 0:
        return []

    # priority: if user passed val_start_name use it
    if args.val_start_name:
        name = args.val_start_name
        if name not in all_files:
            # try with/without extension
            base = os.path.splitext(name)[0]
            matches = [f for f in all_files if os.path.splitext(f)[0] == base]
            if matches:
                start_idx = all_files.index(matches[0])
            else:
                raise FileNotFoundError(f"val_start_name {name} not found in images_dir ({images_dir})")
        else:
            start_idx = all_files.index(name)
    else:
        # else use index
        start_idx = args.val_start if args.val_start is not None else cfg.get('val_start', 0)
        if start_idx < 0:
            start_idx = 0
    val_count = args.val_count if args.val_count is not None else cfg.get('val_count', 20)
    # build list: take val_count files starting from start_idx
    val_list = all_files[start_idx:start_idx + val_count]
    return val_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--attack', choices=['none','fgsm','pgd','bim','mim'], default='none')
    parser.add_argument('--eps', type=float, default=4/255)
    # new attack hyperparams
    parser.add_argument('--attack_iters', type=int, default=None, help='number of iterations for iterative attacks (bim/mim/pgd)')
    parser.add_argument('--attack_alpha', type=float, default=None, help='step size per iteration (if not set uses eps/iters)')
    parser.add_argument('--attack_decay', type=float, default=None, help='momentum decay for MIM (default from config or 1.0)')
    parser.add_argument('--val_start', type=int, default=None, help='start index (0-based) in images_dir for val')
    parser.add_argument('--val_start_name', type=str, default=None, help='start filename (e.g. ISIC_0000101.jpg) for val')
    parser.add_argument('--val_count', type=int, default=None, help='how many val samples to evaluate')

    # defense args
    parser.add_argument('--defense', choices=['none','denoise','tta'], default='none', help='Enable inference defense')
    parser.add_argument('--defense_bits', type=int, default=6, help='bit depth for defense')
    parser.add_argument('--tta_transforms', type=int, default=3, help='num transforms for TTA')

    args = parser.parse_args()

    # load config
    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')

    # build val_list
    try:
        val_list = build_val_list(cfg, args)
    except Exception as e:
        print("Error while building val list:", e)
        raise

    if not val_list:
        print("No validation files found. Check configs/config_isic.json images_dir or set use_splits or val_start/val_start_name.")
        # create empty csv for consistency and exit
        out_csv = os.path.join(cfg.get('out_dir','outputs'), f'eval_{args.attack}_eps{args.eps:.4f}.csv')
        os.makedirs(cfg.get('out_dir','outputs'), exist_ok=True)
        pd.DataFrame(columns=['dice','iou']).to_csv(out_csv, index=False)
        print("Wrote empty csv:", out_csv)
        sys.exit(0)

    print(f"Evaluating {len(val_list)} samples starting from {val_list[0]} (images_dir={cfg['images_dir']})")

    # dataset and loader
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # load model
    model = load_model(args.ckpt, device, cfg)

    # decide attack hyperparameters (priority: CLI args > config > defaults)
    iters = args.attack_iters if args.attack_iters is not None else cfg.get('attack_iters', None)
    if iters is None:
        # defaults
        if args.attack in ('bim','mim','pgd'):
            iters = cfg.get('attack_iters', 10)
        else:
            iters = 1
    alpha = args.attack_alpha if args.attack_alpha is not None else cfg.get('attack_alpha', None)
    if alpha is None:
        # set alpha = eps / iters by default for iterative attacks
        if iters > 0:
            alpha = args.eps / float(iters)
        else:
            alpha = args.eps
    decay = args.attack_decay if args.attack_decay is not None else cfg.get('attack_decay', 1.0)

    print(f"Attack config: attack={args.attack}, eps={args.eps}, iters={iters}, alpha={alpha}, decay={decay}")

    # prepare defense utilities (same as single image)
    defense_mode = args.defense
    def prepare_transforms(n_transforms):
        transforms = []
        transforms.append(_make_identity_transform())
        if n_transforms >= 2:
            transforms.append(_make_flip_transform('h'))
        if n_transforms >= 3:
            transforms.append(_make_shift_transform(2,0))
        return transforms

    if defense_mode != 'none':
        denoise_funcs = [lambda x: median_denoise_tensor(x, kernel_size=3),
                         lambda x: bit_depth_reduce_tensor(x, bits=args.defense_bits)]
        tta_transforms = prepare_transforms(args.tta_transforms)
    else:
        denoise_funcs = None
        tta_transforms = None

    results = []
    loss_ce = torch.nn.CrossEntropyLoss()
    for img, mask in tqdm(val_loader, total=len(val_loader)):
        img_orig = img.clone()
        mask_np = mask.numpy()[0]

        if args.attack == 'none':
            img_in = img_orig
        elif args.attack == 'fgsm':
            img_in = fgsm_attack_on_segmentation(model, img_orig, mask, args.eps, lambda x,y: loss_ce(x,y.to(device)), device)
        elif args.attack == 'pgd':
            img_in = pgd_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters, lambda x,y: loss_ce(x,y.to(device)), device)
        elif args.attack == 'bim':
            img_in = bim_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters, lambda x,y: loss_ce(x,y.to(device)), device)
        elif args.attack == 'mim':
            img_in = mim_attack_on_segmentation(model, img_orig, mask, args.eps, alpha, iters, lambda x,y: loss_ce(x,y.to(device)), device, decay=decay)
        else:
            raise ValueError(f"Unknown attack: {args.attack}")

        # use defense at inference if requested
        if defense_mode == 'none':
            with torch.no_grad():
                logits = model(img_in.to(device))
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
        else:
            agg_pred, cons = tta_predict(model, img_in.cpu(), device, tta_transforms, denoise_funcs=denoise_funcs,
                                         aggregate='vote')
            pred = agg_pred.cpu().numpy()
            # optional: print per-sample consistency
            # print(f"[DEFENSE] sample cons: {cons:.4f}")

        dice = dice_score_numpy(pred, mask_np)
        iou = iou_score_numpy(pred, mask_np)
        results.append({'dice': float(dice), 'iou': float(iou), 'defense': defense_mode})

    df = pd.DataFrame(results)
    if df.empty:
        print("No results were computed (empty dataframe). Exiting.")
    else:
        print(df.describe())

    out_csv = os.path.join(cfg.get('out_dir','outputs'), f'eval_{args.attack}_eps{args.eps:.4f}_defense_{defense_mode}.csv')
    os.makedirs(cfg.get('out_dir','outputs'), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Saved', out_csv)
