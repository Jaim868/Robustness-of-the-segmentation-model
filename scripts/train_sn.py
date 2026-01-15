# scripts/train_sn.py
import os, json, argparse, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime, csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from utils.dataset import ISICDataset
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation
from utils.metrics import dice_score_numpy


def train_sn(cfg_path, alpha=0.7, out_dir='outputs/sn_train'):
    """
    Adversarial Training with Spectral Normalization
    alpha: 0.7 (Based on previous Sweet Spot analysis)
    """
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    # 记录参数
    print(f"--- Starting Spectral Norm Adversarial Training ---")
    print(f"Alpha (Adv Weight): {alpha}")
    print(f"Device: {device}")

    log_csv = os.path.join(out_dir, f'log_sn_alpha_{alpha}.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_clean_dice', 'val_robust_dice', 'timestamp'])

    # Dataset Setup (Same as before)
    import glob
    imgs = sorted(glob.glob(os.path.join(cfg['images_dir'], 'ISIC_*.jpg')))
    imgs = [os.path.basename(x) for x in imgs]
    split_idx = int(len(imgs) * 0.8)
    train_list = imgs[:split_idx]
    val_list = imgs[split_idx:split_idx + 50]  # Fast validation

    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- 关键点：初始化开启 Spectral Norm 的模型 ---
    model = UNet(
        in_channels=cfg['in_channels'],
        n_classes=cfg['num_classes'],
        base_ch=cfg.get('base_ch', 16),
        spectral_norm=True  # <--- Enable Spectral Normalization
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    ce = nn.CrossEntropyLoss()

    # PGD Params
    adv_eps = cfg.get('adv_eps', 4 / 255)
    adv_iters = cfg.get('adv_iters', 3)
    adv_alpha = adv_eps / adv_iters

    best_robust = 0.0

    for epoch in range(cfg.get('num_epochs', 20)):
        model.train()
        running_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch + 1} SN-Adv", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # 1. Clean Loss
            logits_clean = model(imgs)
            loss_clean = ce(logits_clean, masks)

            # 2. Adversarial Sample Generation
            # SN 限制了 Lipschitz 常数，理论上这会让梯度更加稳定，攻击生成也会更规范
            model.eval()
            adv_imgs = pgd_attack_on_segmentation(
                model, imgs, masks,
                eps=adv_eps, alpha=adv_alpha, iters=adv_iters,
                loss_fn=ce, device=device
            )
            model.train()

            # 3. Adversarial Loss
            logits_adv = model(adv_imgs)
            loss_adv = ce(logits_adv, masks)

            # 4. Combined Weighted Loss
            loss = (1.0 - alpha) * loss_clean + alpha * loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        clean_dices = []
        robust_dices = []

        # Eval params (Stronger attack for validation)
        eval_iters = 10
        eval_alpha = adv_eps / 10

        for v_img, v_mask in val_loader:
            v_img = v_img.to(device)
            mask_np = v_mask.numpy()[0]
            v_mask_dev = v_mask.to(device)

            # Clean
            with torch.no_grad():
                l_clean = model(v_img)
                pred_clean = torch.argmax(torch.softmax(l_clean, dim=1), dim=1).cpu().numpy()[0]
                clean_dices.append(dice_score_numpy(pred_clean, mask_np))

            # Robust (PGD-10)
            adv_v_img = pgd_attack_on_segmentation(
                model, v_img, v_mask_dev,
                eps=adv_eps, alpha=eval_alpha, iters=eval_iters,
                loss_fn=ce, device=device
            )
            with torch.no_grad():
                l_adv = model(adv_v_img)
                pred_adv = torch.argmax(torch.softmax(l_adv, dim=1), dim=1).cpu().numpy()[0]
                robust_dices.append(dice_score_numpy(pred_adv, mask_np))

        avg_clean = np.mean(clean_dices)
        avg_robust = np.mean(robust_dices)
        avg_loss = running_loss / len(train_loader)

        print(f"[Ep {epoch + 1}] Loss: {avg_loss:.4f} | Clean Dice: {avg_clean:.4f} | Robust Dice: {avg_robust:.4f}")

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_loss, avg_clean, avg_robust, datetime.datetime.now()])

        # Save Best Model based on Robust Dice
        if avg_robust > best_robust:
            best_robust = avg_robust
            torch.save(model.state_dict(), os.path.join(out_dir, 'unet_sn_best.pth'))
            print("  >>> Best Robust Model Saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for adversarial loss (recommend 0.7 or 0.5)')
    parser.add_argument('--out_dir', default='outputs/sn_train')
    args = parser.parse_args()

    train_sn(args.config, args.alpha, args.out_dir)