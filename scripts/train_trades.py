# scripts/train_trades.py
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
from utils.metrics import dice_score_numpy
from scripts.trades import trades_loss  # 导入刚才写的函数
# 还需要导入普通 PGD 用于验证集评估
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation


def train_trades(cfg_path, beta, out_dir):
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- Starting TRADES Training ---")
    print(f"Beta (Robust Weight): {beta}")

    # 日志
    log_csv = os.path.join(out_dir, f'log_trades_beta_{beta}.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'total_loss', 'clean_loss', 'robust_loss', 'val_clean_dice', 'val_robust_dice', 'timestamp'])

    # 数据集
    import glob
    imgs = sorted(glob.glob(os.path.join(cfg['images_dir'], 'ISIC_*.jpg')))
    imgs = [os.path.basename(x) for x in imgs]
    split_idx = int(len(imgs) * 0.8)
    train_list = imgs[:split_idx]
    val_list = imgs[split_idx:split_idx + 50]

    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 模型：建议开启 Spectral Norm (sn=True) 配合 TRADES
    model = UNet(
        in_channels=cfg['in_channels'],
        n_classes=cfg['num_classes'],
        base_ch=cfg.get('base_ch', 16),  # 注意改成你的 base_ch，比如 8
        spectral_norm=True
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # TRADES 参数
    adv_eps = cfg.get('adv_eps', 4 / 255)
    adv_iters = 5  # TRADES 训练通常比普通 AT 慢，步数可以设小一点 (如 3-5) 或者使用 Free-TRADES
    adv_step_size = adv_eps / adv_iters * 1.5  # 步长通常稍大于 eps/iters

    best_robust = 0.0

    for epoch in range(cfg.get('num_epochs', 20)):
        model.train()

        # Metrics trackers
        epoch_total_loss = 0
        epoch_clean_loss = 0
        epoch_robust_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch + 1} TRADES", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            opt.zero_grad()

            # --- TRADES LOSS CALCULATION ---
            loss, l_clean, l_robust = trades_loss(
                model=model,
                x_natural=imgs,
                y=masks,
                optimizer=opt,
                step_size=adv_step_size,
                epsilon=adv_eps,
                perturb_steps=adv_iters,
                beta=beta,
                distance='l_inf'
            )

            loss.backward()
            opt.step()

            epoch_total_loss += loss.item()
            epoch_clean_loss += l_clean
            epoch_robust_loss += l_robust

        # Normalize metrics
        n_batches = len(train_loader)
        epoch_total_loss /= n_batches
        epoch_clean_loss /= n_batches
        epoch_robust_loss /= n_batches

        # Validation (Standard Eval)
        model.eval()
        clean_dices = []
        robust_dices = []
        ce_loss = nn.CrossEntropyLoss()

        # Val params (Standard PGD-10 for evaluation)
        val_eps = adv_eps
        val_iters = 10
        val_alpha = val_eps / 10

        for v_img, v_mask in val_loader:
            v_img = v_img.to(device)
            mask_np = v_mask.numpy()[0]
            v_mask_dev = v_mask.to(device)

            # Clean
            with torch.no_grad():
                l_clean = model(v_img)
                pred_clean = torch.argmax(torch.softmax(l_clean, dim=1), dim=1).cpu().numpy()[0]
                clean_dices.append(dice_score_numpy(pred_clean, mask_np))

            # Robust (Standard PGD attack on CrossEntropy, NOT TRADES attack, for fair comparison)
            adv_v_img = pgd_attack_on_segmentation(
                model, v_img, v_mask_dev,
                eps=val_eps, alpha=val_alpha, iters=val_iters,
                loss_fn=ce_loss, device=device
            )
            with torch.no_grad():
                l_adv = model(adv_v_img)
                pred_adv = torch.argmax(torch.softmax(l_adv, dim=1), dim=1).cpu().numpy()[0]
                robust_dices.append(dice_score_numpy(pred_adv, mask_np))

        avg_clean = np.mean(clean_dices)
        avg_robust = np.mean(robust_dices)

        print(
            f"[Ep {epoch + 1}] Total: {epoch_total_loss:.3f} (Cln:{epoch_clean_loss:.3f}/Rob:{epoch_robust_loss:.3f}) | Val Dice: C={avg_clean:.3f}/R={avg_robust:.3f}")

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch + 1, epoch_total_loss, epoch_clean_loss, epoch_robust_loss, avg_clean, avg_robust,
                 datetime.datetime.now()])

        if avg_robust > best_robust:
            best_robust = avg_robust
            torch.save(model.state_dict(), os.path.join(out_dir, f'unet_trades_beta{beta}_best.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    # 论文推荐 Beta 1.0 到 6.0。对于分割任务，通常 1.0 到 3.0 比较合适，太大很难收敛
    parser.add_argument('--beta', type=float, default=1.0, help='Regularization parameter beta')
    parser.add_argument('--base_ch', type=int, default=8, help='Base Channel')
    parser.add_argument('--out_dir', default='outputs/trades_train')
    args = parser.parse_args()

    # 临时覆盖 config 中的 base_ch
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['base_ch'] = args.base_ch
    # 这里有点 hacky，但为了方便脚本传递
    # 实际项目中建议直接用 ArgumentParser 参数覆盖

    train_trades(args.config, args.beta, args.out_dir)