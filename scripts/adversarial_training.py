# scripts/adversarial_training.py
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



def train_adv(cfg_path, out_dir=None):
    with open(cfg_path) as f:
        cfg = json.load(f)
    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    out_dir = out_dir if out_dir else cfg.get('out_dir','outputs')
    os.makedirs(out_dir, exist_ok=True)

    # dataset (use use_splits or direct)
    use_splits = cfg.get('use_splits', False)
    if use_splits:
        with open('splits.json') as f:
            splits = json.load(f)
        train_list = splits.get('train', [])
        val_list = splits.get('val', [])
    else:
        import glob
        # imgs = sorted(list(set(glob.glob(os.path.join(cfg['images_dir'], 'ISIC_*.jpg')))))
        # 修改为 (支持更多格式)
        imgs = sorted(glob.glob(os.path.join(cfg['images_dir'], '*.*')))
        # 过滤非图片文件
        imgs = [x for x in imgs if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        imgs = [os.path.basename(x) for x in imgs]
        start = cfg.get('train_start',0)
        limit = cfg.get('train_limit', None)
        if limit is None:
            train_list = imgs[start:]
        else:
            train_list = imgs[start:start+limit]
        remaining = [f for f in imgs if f not in train_list]
        val_list = remaining[:cfg.get('val_count',20)]

    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers',0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_channels=cfg['in_channels'], n_classes=cfg['num_classes'], base_ch=cfg.get('base_ch',16)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    ce = nn.CrossEntropyLoss()


    # PGD attack hyperparams for adversarial training
    adv_eps = cfg.get('adv_eps', 4 / 255)
    adv_iters = int(cfg.get('adv_iters', 1))
    if adv_iters <= 0:
        adv_iters = 1

    # robust adv_alpha: prefer explicit numeric in config, otherwise default to adv_eps/adv_iters
    adv_alpha = cfg.get('adv_alpha', None)
    if adv_alpha is None:
        adv_alpha = float(adv_eps) / float(adv_iters)
    else:
        adv_alpha = float(adv_alpha)


    best_val = 0.0
    metrics_csv = os.path.join(out_dir, 'adv_train_metrics.csv')
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','train_loss','val_dice','timestamp'])

    for epoch in range(cfg.get('num_epochs', 30)):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} adv-train"):
            # # debug: 单步验证模型对输入是否有梯度
            # imgs0, masks0 = imgs[:1].to(device), masks[:1].to(device)
            # if masks0.ndim == 4 and masks0.shape[1] == 1:
            #     masks0 = masks0.squeeze(1)
            # masks0 = masks0.long()
            # imgs0 = imgs0.detach().clone()
            # imgs0.requires_grad_(True)
            # logits0 = model(imgs0)
            # loss0 = ce(logits0, masks0)
            # gr = None
            # try:
            #     loss0.backward()
            #     gr = imgs0.grad
            # except Exception:
            #     try:
            #         gr = torch.autograd.grad(loss0, imgs0)[0]
            #     except Exception as e:
            #         print("Gradient debug failed:", e)
            # print("Debug grad is None?", gr is None, "shape:", None if gr is None else gr.shape, "norm:",
            #       None if gr is None else gr.norm().item())
            # # end debug

            imgs = imgs.to(device); masks = masks.to(device)

            # free cached memory before PGD loop
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # generate PGD adversarial examples on-the-fly

            adv_imgs = pgd_attack_on_segmentation(model, imgs, masks, adv_eps, adv_alpha, adv_iters, lambda x,y: ce(x,y), device)
            # compute loss on adversarial images (Madry-style)

            # optional: free cached memory again
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            logits = model(adv_imgs)
            loss = ce(logits, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()

        # validation clean
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks_np = masks.numpy()
                logits = model(imgs)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
                for p,g in zip(preds, masks_np):
                    dices.append( (2*(p & g).sum())/ (p.sum()+g.sum()+1e-6) if (p.sum()+g.sum())>0 else 0.0)
        val_dice = float(np.mean(dices)) if len(dices)>0 else 0.0
        # log
        with open(metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, running_loss/len(train_loader), val_dice, datetime.datetime.now().isoformat()])
        # save best
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, 'unet_adv_trained.pth'))
        print(f"[Adv-Train] Epoch {epoch+1} train_loss={running_loss/len(train_loader):.4f}, val_dice={val_dice:.4f}")

    print("Adversarial training finished. Best val dice:", best_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    train_adv(args.config, out_dir=args.out)
