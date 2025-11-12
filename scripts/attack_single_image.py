# scripts/attack_single_image.py
import os, json, argparse, logging, datetime, csv, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
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

def load_cfg(cfg_path):
    with open(cfg_path) as f:
        return json.load(f)

def load_model(ckpt_path, cfg, device):
    base_ch = cfg.get('base_ch', 16)
    model = UNet(in_channels=cfg.get('in_channels',3), n_classes=cfg.get('num_classes',2), base_ch=base_ch).to(device)

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

    new_sd = {}
    for k, v in sd.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_sd[new_key] = v

    model.load_state_dict(new_sd)
    model.eval()
    return model

def pil_to_tensor(img_pil):
    arr = np.array(img_pil).astype('float32')/255.0
    if arr.ndim==2:
        arr = np.stack([arr,arr,arr], axis=-1)
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
    return t

def tensor_to_mask_img(mask_tensor):
    if isinstance(mask_tensor, torch.Tensor):
        arr = mask_tensor.cpu().numpy()
    else:
        arr = mask_tensor
    if arr.ndim==3:
        arr = arr[0]
    img = (arr.astype(np.uint8))*255
    return Image.fromarray(img)

def save_vis(orig_img_np, perturb, pred_clean, pred_adv_no_def, pred_adv_def, out_prefix):
    """
    Save 4- or 5-panel visualization:
    panels: Original | Perturbation x10 | Clean overlay | Adv no-def overlay | Adv defended overlay (optional)
    """
    # ensure perturb detached on CPU
    if isinstance(perturb, torch.Tensor):
        perturb = perturb.detach().cpu()
    pert_np = perturb.squeeze(0).permute(1,2,0).cpu().numpy()
    pert_vis = (pert_np * 10.0) + 0.5  # amplify
    # determine number of panels
    panels = 4 + (1 if pred_adv_def is not None else 0)
    figsize = (4*panels, 4)
    fig, axes = plt.subplots(1, panels, figsize=figsize)
    if panels == 1:
        axes = [axes]
    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original')
    axes[1].imshow(np.clip(pert_vis,0,1))
    axes[1].set_title('Perturbation x10')
    axes[2].imshow(orig_img_np)
    axes[2].imshow(pred_clean, alpha=0.5, cmap='Reds')
    axes[2].set_title('Clean Pred Overlay')
    axes[3].imshow(orig_img_np)
    axes[3].imshow(pred_adv_no_def, alpha=0.5, cmap='Reds')
    axes[3].set_title('Adv (no defense) Overlay')
    if pred_adv_def is not None:
        axes[4].imshow(orig_img_np)
        axes[4].imshow(pred_adv_def, alpha=0.5, cmap='Reds')
        axes[4].set_title('Adv (defended) Overlay')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_prefix + '_vis.png', bbox_inches='tight')
    plt.close(fig)

def prepare_transforms(n_transforms):
    transforms = []
    transforms.append(_make_identity_transform())
    if n_transforms >= 2:
        transforms.append(_make_flip_transform('h'))
    if n_transforms >= 3:
        transforms.append(_make_shift_transform(2,0))
    return transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--image', required=True, help='image filename under images_dir or full path')
    parser.add_argument('--attack', choices=['fgsm','pgd','bim','mim','none'], default='fgsm')
    parser.add_argument('--eps', type=float, default=0.0157)
    parser.add_argument('--attack_iters', type=int, default=None, help='iters for iterative attacks')
    parser.add_argument('--attack_alpha', type=float, default=None, help='step size per iter (defaults to eps/iters)')
    parser.add_argument('--attack_decay', type=float, default=None, help='momentum decay for MIM')
    parser.add_argument('--defense', choices=['none','denoise','tta'], default='none', help='Enable input defense at inference: none/denoise/tta')
    parser.add_argument('--defense_bits', type=int, default=6, help='bit depth reduction when denoise/tta is used (1..8)')
    parser.add_argument('--tta_transforms', type=int, default=3, help='number of TTA transforms to use (recommended 3)')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    images_dir = cfg.get('images_dir')
    masks_dir = cfg.get('masks_dir')
    out_dir = args.out_dir if args.out_dir else cfg.get('out_dir','outputs')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, cfg, device)

    # resolve image path
    img_path = args.image
    if os.path.isabs(img_path):
        img_path = os.path.normpath(img_path)
    else:
        if os.path.dirname(img_path):
            img_path = os.path.normpath(os.path.join(os.getcwd(), img_path))
        else:
            img_path = os.path.join(images_dir, img_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_pil = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_pil.size
    # save original image copy
    name = os.path.splitext(os.path.basename(img_path))[0]
    img_pil.save(os.path.join(out_dir, f'single_{name}_orig.png'))

    img_t = pil_to_tensor(img_pil).to(device)
    orig_np = np.array(img_pil).astype('float32')/255.0

    # try find mask
    base = name
    mask_candidates = [
        os.path.join(masks_dir, base + '_segmentation.png'),
        os.path.join(masks_dir, base + '.png'),
        os.path.join(masks_dir, base + '_mask.png')
    ]
    gt_mask_path = None
    for p in mask_candidates:
        if os.path.exists(p):
            gt_mask_path = p
            break
    if gt_mask_path:
        gt_pil = Image.open(gt_mask_path).convert('L').resize((orig_w, orig_h))
        gt_np = (np.array(gt_pil) > 127).astype('uint8')
        has_gt = True
    else:
        gt_np = None
        has_gt = False

    # prepare defense utilities
    defense_mode = args.defense
    if defense_mode != 'none':
        denoise_funcs = [
            lambda x: median_denoise_tensor(x, kernel_size=3),
            lambda x: bit_depth_reduce_tensor(x, bits=args.defense_bits)
        ]
        tta_transforms = prepare_transforms(args.tta_transforms)
    else:
        denoise_funcs = None
        tta_transforms = None

    # Predict clean (optionally with defense)
    if defense_mode == 'none':
        with torch.no_grad():
            logits = model(img_t.to(device))
            probs = torch.softmax(logits, dim=1)
            pred_clean = torch.argmax(probs, dim=1).cpu().numpy()[0]
    else:
        agg_pred, consistency = tta_predict(model, img_t.cpu(), device, transforms=tta_transforms, denoise_funcs=denoise_funcs, aggregate='soft')
        pred_clean = agg_pred.cpu().numpy()
        print(f"[DEFENSE] TTA consistency (clean): {consistency:.4f}")

    # choose label for attack
    if has_gt:
        label_t = torch.from_numpy(gt_np).unsqueeze(0).to(device)
    else:
        label_t = torch.from_numpy(pred_clean).unsqueeze(0).to(device)

    # attack hyperparams
    iters = args.attack_iters if args.attack_iters is not None else cfg.get('attack_iters', None)
    if iters is None:
        iters = cfg.get('attack_iters', 10) if args.attack in ('bim','mim','pgd') else 1
    alpha = args.attack_alpha if args.attack_alpha is not None else cfg.get('attack_alpha', None)
    if alpha is None:
        alpha = args.eps / float(iters) if iters>0 else args.eps
    decay = args.attack_decay if args.attack_decay is not None else cfg.get('attack_decay', 1.0)

    print(f"Single-image attack config: attack={args.attack}, eps={args.eps}, iters={iters}, alpha={alpha}, decay={decay}, defense={defense_mode}")

    # perform attack -> adv (ensure adv is detached)
    if args.attack == 'none':
        adv = img_t.detach()
    elif args.attack == 'fgsm':
        adv = fgsm_attack_on_segmentation(model, img_t, label_t, args.eps, lambda x,y: torch.nn.CrossEntropyLoss()(x,y.to(device)), device).detach()
    elif args.attack == 'pgd':
        adv = pgd_attack_on_segmentation(model, img_t, label_t, args.eps, alpha, iters, lambda x,y: torch.nn.CrossEntropyLoss()(x,y.to(device)), device).detach()
    elif args.attack == 'bim':
        adv = bim_attack_on_segmentation(model, img_t, label_t, args.eps, alpha, iters, lambda x,y: torch.nn.CrossEntropyLoss()(x,y.to(device)), device).detach()
    elif args.attack == 'mim':
        adv = mim_attack_on_segmentation(model, img_t, label_t, args.eps, alpha, iters, lambda x,y: torch.nn.CrossEntropyLoss()(x,y.to(device)), device, decay=decay).detach()
    else:
        raise ValueError(f"Unknown attack: {args.attack}")

    # save adv image (raw adversarial image)
    adv_img_np = adv.squeeze(0).permute(1,2,0).cpu().numpy()
    adv_img_pil = Image.fromarray((np.clip(adv_img_np,0,1)*255).astype('uint8'))
    adv_img_pil.save(os.path.join(out_dir, f'single_{name}_adv.png'))  # adversarial input (no defense)

    # Predict adv without defense (no defense inference)
    with torch.no_grad():
        logits_adv_no_def = model(adv.to(device))
        pred_adv_no_def = torch.argmax(torch.softmax(logits_adv_no_def, dim=1), dim=1).cpu().numpy()[0]

    # Predict adv with defense (if requested) and also prepare a representative defended input image to save
    pred_adv_def = None
    defended_input_saved = None
    if defense_mode == 'none':
        # no defended prediction
        pred_adv_def = None
    elif defense_mode == 'denoise':
        # apply first denoise function to adv (representative) and save that as defended input
        adv_cpu = adv.cpu()
        defended_input = denoise_funcs[0](adv_cpu)  # tensor 1xCxHxW
        defended_input_np = defended_input.squeeze(0).permute(1,2,0).cpu().numpy()
        defended_input_pil = Image.fromarray((np.clip(defended_input_np,0,1)*255).astype('uint8'))
        defended_input_pil.save(os.path.join(out_dir, f'single_{name}_adv_defended_input.png'))
        defended_input_saved = os.path.join(out_dir, f'single_{name}_adv_defended_input.png')
        # model prediction on defended input: we can use tta_predict with transforms=identity to be consistent
        agg_pred_def, cons_def = tta_predict(model, defended_input.cpu(), device, transforms=[_make_identity_transform()], denoise_funcs=None, aggregate='soft')
        pred_adv_def = agg_pred_def.cpu().numpy()
        print(f"[DEFENSE] defended consistency: {cons_def:.4f}")
    else:  # 'tta'
        # For tta mode, we compute agg_pred over transforms/denoises and save a representative denoised input (first denoise + identity transform)
        adv_cpu = adv.cpu()
        # representative defended input
        rep_def_input = denoise_funcs[0](adv_cpu)
        rep_def_np = rep_def_input.squeeze(0).permute(1,2,0).cpu().numpy()
        rep_def_pil = Image.fromarray((np.clip(rep_def_np,0,1)*255).astype('uint8'))
        rep_def_pil.save(os.path.join(out_dir, f'single_{name}_adv_defended_input.png'))
        defended_input_saved = os.path.join(out_dir, f'single_{name}_adv_defended_input.png')
        # get aggregated defended prediction via TTA
        agg_pred_def, cons_def = tta_predict(model, adv_cpu, device, transforms=tta_transforms, denoise_funcs=denoise_funcs, aggregate='soft')
        pred_adv_def = agg_pred_def.cpu().numpy()
        print(f"[DEFENSE] TTA defended consistency: {cons_def:.4f}")

    # Save split predictions and vis
    pred_clean_img = tensor_to_mask_img(pred_clean)
    pred_adv_no_def_img = tensor_to_mask_img(pred_adv_no_def)
    if pred_adv_def is not None:
        pred_adv_def_img = tensor_to_mask_img(pred_adv_def)
    else:
        pred_adv_def_img = None

    pred_clean_img.save(os.path.join(out_dir, f'single_{name}_pred.png'))
    pred_adv_no_def_img.save(os.path.join(out_dir, f'single_{name}_pred_adv_no_def.png'))
    if pred_adv_def_img is not None:
        pred_adv_def_img.save(os.path.join(out_dir, f'single_{name}_pred_adv_def.png'))

    # perturbation and vis (include defended pred if available)
    perturb = (adv - img_t.to(device)).detach().cpu()
    save_vis(orig_np, perturb, pred_clean, pred_adv_no_def, pred_adv_def, os.path.join(out_dir, f'single_{name}'))

    # compute metrics if GT available: compute for clean, adv_no_def, adv_def (if available)
    dice = iou = dice_adv = iou_adv = dice_adv_def = iou_adv_def = None
    if has_gt:
        dice = float(dice_score_numpy(pred_clean, gt_np))
        iou = float(iou_score_numpy(pred_clean, gt_np))
        dice_adv = float(dice_score_numpy(pred_adv_no_def, gt_np))
        iou_adv = float(iou_score_numpy(pred_adv_no_def, gt_np))
        if pred_adv_def is not None:
            dice_adv_def = float(dice_score_numpy(pred_adv_def, gt_np))
            iou_adv_def = float(iou_score_numpy(pred_adv_def, gt_np))

    # append to single_image_results.csv (add defended fields)
    csv_path = os.path.join(out_dir, 'single_image_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as cf:
        w = csv.writer(cf)
        if write_header:
            w.writerow(['image','has_gt','dice_clean','iou_clean','dice_adv_no_def','iou_adv_no_def','dice_adv_def','iou_adv_def','attack','eps','iters','alpha','decay','defense','defense_bits','defended_input_path','timestamp'])
        w.writerow([name, has_gt, dice, iou, dice_adv, iou_adv, dice_adv_def, iou_adv_def, args.attack, args.eps, iters, alpha, decay, defense_mode, args.defense_bits, defended_input_saved, datetime.datetime.now().isoformat()])

    print("Saved: orig, pred, adv, defended input (if any), vis and csv to", out_dir)
    if has_gt:
        print(f"Metrics (clean): Dice={dice:.4f}, IoU={iou:.4f}; (adv no def) Dice={dice_adv:.4f}, IoU={iou_adv:.4f}")
        if dice_adv_def is not None:
            print(f"(adv defended) Dice={dice_adv_def:.4f}, IoU={iou_adv_def:.4f}")
    else:
        print("No GT found; results saved but metrics are not computed (used model pred as pseudo-label for attack).")

if __name__ == '__main__':
    main()