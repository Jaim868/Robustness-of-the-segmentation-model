# scripts/train_tradeoff.py
import os, json, argparse, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime, csv

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from utils.dataset import ISICDataset
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation
from utils.metrics import dice_score_numpy


def train_tradeoff(cfg_path, alpha, out_dir):
    """
    alpha: float, 0.0 to 1.0
           0.0 = Pure Clean Training
           1.0 = Pure Adversarial Training
           0.5 = Balanced
    """
    with open(cfg_path) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)

    # 日志文件
    log_csv = os.path.join(out_dir, f'log_alpha_{alpha:.2f}.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_clean_dice', 'val_robust_dice', 'alpha'])

    # 1. Dataset & Loader
    # 简化的读取逻辑，你也可以用 splits.json
    import glob
    imgs = sorted(glob.glob(os.path.join(cfg['images_dir'], 'ISIC_*.jpg')))
    imgs = [os.path.basename(x) for x in imgs]

    # 简单的划分: 前80%训练，后20%验证
    split_idx = int(len(imgs) * 0.8)
    train_list = imgs[:split_idx]
    val_list = imgs[split_idx:split_idx + 50]  # 验证集取50个以加快Robust Eval速度

    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 2. Model
    model = UNet(in_channels=cfg['in_channels'], n_classes=cfg['num_classes'], base_ch=cfg.get('base_ch', 16)).to(
        device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    ce = nn.CrossEntropyLoss()

    # Attack Params for Training
    adv_eps = cfg.get('adv_eps', 4 / 255)
    adv_iters = cfg.get('adv_iters', 3)  # 训练时步数少一点以加快速度
    adv_alpha = adv_eps / adv_iters

    print(f"--- Starting Trade-off Training | Alpha={alpha} ---")

    for epoch in range(cfg.get('num_epochs', 20)):
        model.train()
        running_loss = 0.0

        # Training Loop
        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch + 1} A={alpha}", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # A. Clean Loss
            logits_clean = model(imgs)
            loss_clean = ce(logits_clean, masks)

            # B. Adversarial Loss (只有当 alpha > 0 时才计算)
            if alpha > 0.0:
                model.eval()  # 生成对抗样本时通常设为eval模式或保持train都可以，这里用eval避免BN统计偏移
                adv_imgs = pgd_attack_on_segmentation(
                    model, imgs, masks,
                    eps=adv_eps, alpha=adv_alpha, iters=adv_iters,
                    loss_fn=ce, device=device
                )
                model.train()

                logits_adv = model(adv_imgs)
                loss_adv = ce(logits_adv, masks)
            else:
                loss_adv = torch.tensor(0.0).to(device)

            # C. Combined Loss
            # L = (1 - alpha) * L_clean + alpha * L_adv
            loss = (1.0 - alpha) * loss_clean + alpha * loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # Validation Loop (Clean & Robust)
        model.eval()
        clean_dices = []
        robust_dices = []

        # 验证时攻击参数 (使用标准的 PGD-10 进行评估)
        eval_iters = 10
        eval_alpha = adv_eps / 10

        for v_img, v_mask in val_loader:
            v_img = v_img.to(device)
            mask_np = v_mask.numpy()[0]
            v_mask_dev = v_mask.to(device)

            # 1. Clean Eval
            with torch.no_grad():
                l_clean = model(v_img)
                pred_clean = torch.argmax(torch.softmax(l_clean, dim=1), dim=1).cpu().numpy()[0]
                clean_dices.append(dice_score_numpy(pred_clean, mask_np))

            # 2. Robust Eval (Generate PGD-10 attack)
            # 只有在非纯Clean训练时，或者为了画图对比时才跑这个
            # 这里为了Trade-off分析，我们总是跑这个
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
            csv.writer(f).writerow([epoch + 1, avg_loss, avg_clean, avg_robust, alpha])

    # 保存最终模型
    model_path = os.path.join(out_dir, f'unet_alpha_{alpha:.2f}.pth')
    torch.save(model.state_dict(), model_path)
    return avg_clean, avg_robust


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for adversarial loss (0.0 - 1.0)')
    parser.add_argument('--out_dir', default='outputs/tradeoff')
    args = parser.parse_args()

    train_tradeoff(args.config, args.alpha, args.out_dir)