# scripts/train_unet.py
import csv
import datetime
import logging
import os, sys, json, random, argparse
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# project_root = parent directory of the scripts directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------------------
from models.unet import UNet
from utils.dataset import ISICDataset
from utils.metrics import dice_score_numpy
# scripts/train_unet.py 中引入新模块
from utils.robust_aug import AddGaussianNoise, mixup_data, mixup_criterion

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(log_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    # remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def train_one_epoch(model, loader, opt, loss_fn, device, mixup_args=None, noise_injector=None):
    model.train()
    running = 0.0
    n = 0

    use_mixup = mixup_args.get('use', False) if mixup_args else False
    mixup_alpha = mixup_args.get('alpha', 1.0) if mixup_args else 1.0

    for imgs, masks in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        # --- A. 高斯噪声注入 ---
        if noise_injector is not None:
            # 在输入模型前，给图像加噪
            imgs = noise_injector(imgs)

        opt.zero_grad()

        # --- B. Mixup 逻辑 ---
        if use_mixup:
            # 生成混合样本
            imgs_mixed, masks_a, masks_b, lam = mixup_data(imgs, masks, alpha=mixup_alpha, device=device)

            # 前向传播 (使用混合后的图像)
            logits = model(imgs_mixed)

            # 计算 Mixup Loss (根据 lambda 加权两个标签的 loss)
            loss = mixup_criterion(loss_fn, logits, masks_a, masks_b, lam)

        else:
            # --- 正常训练逻辑 ---
            logits = model(imgs)
            loss = loss_fn(logits, masks)

        loss.backward()
        opt.step()

        running += loss.item()
        n += 1

    return (running / n) if n > 0 else 0.0

def eval_model(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for p, g in zip(preds, masks_np):
                dices.append(dice_score_numpy(p, g))
    if len(dices) == 0:
        return 0.0, 0.0
    return np.mean(dices), np.std(dices)

def main(cfg_path, train_limit=None, train_start=None):
    with open(cfg_path) as f:
        cfg = json.load(f)
    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    out_dir = cfg.get('out_dir', './outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(out_dir, 'train.log')
    logger = setup_logger(log_path)
    logger.info("Starting training, output directory: %s", out_dir)
    # Save config snapshot
    cfg_snapshot_path = os.path.join(out_dir, 'config_used.json')
    with open(cfg_snapshot_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    logger.info("Saved config snapshot to %s", cfg_snapshot_path)

    use_splits = cfg.get('use_splits', False)
    if use_splits:
        # old behavior: read splits.json
        with open('splits.json') as f:
            splits = json.load(f)
        train_list = splits.get('train', [])
        val_list = splits.get('val', [])
    else:
        images_dir = cfg['images_dir']
        patterns = ['ISIC_*.jpg', 'ISIC_*.jpeg', 'ISIC_*.png']
        import glob
        all_files = []
        for p in patterns:
            all_files.extend(glob.glob(os.path.join(images_dir, p)))
        all_files = sorted(list(set(all_files)))
        all_files = [os.path.basename(x) for x in all_files]
        start = train_start if train_start is not None else cfg.get('train_start', 0)
        limit = train_limit if train_limit is not None else cfg.get('train_limit', None)
        if start < 0:
            start = 0
        if limit is None:
            train_list = all_files[start:]
        else:
            train_list = all_files[start:start+limit]
        remaining = [f for f in all_files if f not in train_list]
        val_count = cfg.get('val_count', 100)
        val_list = remaining[:val_count]

    logger.info("Training samples: %d; Validation samples: %d", len(train_list), len(val_list))
    logger.info("First 5 train samples: %s", str(train_list[:5]))

    # Build datasets and loaders
    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers',0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    base_ch = cfg.get('base_ch', 16)
    # 读取 denoise 参数，默认为 False
    use_denoise = cfg.get('denoise', False)

    if use_denoise:
        logger.info("Feature Denoising Module (Non-Local Blocks) is ENABLED.")
    else:
        logger.info("Feature Denoising Module is DISABLED.")

    # 传入 denoise 参数
    model = UNet(
        in_channels=cfg['in_channels'],
        n_classes=cfg['num_classes'],
        base_ch=base_ch,
        denoise=use_denoise
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # loss: CE + soft dice
    ce = nn.CrossEntropyLoss()
    def loss_fn(logits, target):
        l_ce = ce(logits, target)
        probs = torch.softmax(logits, dim=1)[:,1,:,:]
        t = (target==1).float()
        inter = (probs * t).sum(dim=[1,2])
        denom = probs.sum(dim=[1,2]) + t.sum(dim=[1,2])
        dice = 1 - ((2*inter + 1e-6) / (denom + 1e-6)).mean()
        return l_ce + dice

    # Prepare metrics CSV
    metrics_csv = os.path.join(out_dir, 'train_metrics.csv')
    write_header = not os.path.exists(metrics_csv)
    if write_header:
        with open(metrics_csv, 'w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(['epoch','train_loss','val_dice_mean','val_dice_std','timestamp'])

    best_dice = 0.0
    num_epochs = cfg.get('num_epochs', 30)
    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch+1, num_epochs)
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_mean, val_std = eval_model(model, val_loader, device)
        logger.info("Epoch %d: train_loss=%.6f, val_dice=%.6f ± %.6f", epoch+1, tr_loss, val_mean, val_std)

        # append csv
        with open(metrics_csv, 'a', newline='') as cf:
            w = csv.writer(cf)
            w.writerow([epoch+1, tr_loss, val_mean, val_std, datetime.datetime.now().isoformat()])

        # save best
        if val_mean > best_dice:
            best_dice = val_mean
            ckpt_path = os.path.join(out_dir, 'unet_best_denoise.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved new best model to %s (val_dice=%.6f)", ckpt_path, val_mean)

    logger.info("Training finished. Best val dice: %.6f", best_dice)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--train_limit', type=int, default=None, help='If set, limit training set to first N samples read from images_dir')

    parser.add_argument('--train_start', type=int, default=None, help='If set, start index (0-based) in images_dir listing')
    args = parser.parse_args()

    main(args.config, train_limit=args.train_limit, train_start=args.train_start)