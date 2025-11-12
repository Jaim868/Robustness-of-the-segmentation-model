# scripts/train_unet.py
import os, sys, json, random, argparse
import torch
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running = 0.0
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader) if len(loader) > 0 else 0.0

def eval_model(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.numpy()
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for p, g in zip(preds, masks):
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

    use_splits = cfg.get('use_splits', False)  # default False: do not use splits.json

    if use_splits:
        # existing behavior: load splits.json
        with open('splits.json') as f:
            splits = json.load(f)
        train_list = splits.get('train', [])
        val_list = splits.get('val', [])
    else:
        # read images directly from images_dir
        images_dir = cfg['images_dir']
        masks_dir = cfg['masks_dir']
        # find files matching ISIC_*.jpg (done inside ISICDataset, but we need list for slicing)
        patterns = ['ISIC_*.jpg', 'ISIC_*.jpeg', 'ISIC_*.png']
        all_files = []
        import glob
        for p in patterns:
            all_files.extend(glob.glob(os.path.join(images_dir, p)))
        all_files = sorted(list(set(all_files)))
        all_files = [os.path.basename(x) for x in all_files]
        # apply start and limit
        start = train_start if train_start is not None else cfg.get('train_start', 0)
        limit = train_limit if train_limit is not None else cfg.get('train_limit', None)
        if start < 0:
            start = 0
        if limit is None:
            train_list = all_files[start:]
        else:
            train_list = all_files[start:start+limit]
        # val list: use 'val_dir' if provided in config, else use remaining files after train
        if 'val_images_dir' in cfg and cfg.get('val_images_dir'):
            val_images_dir = cfg['val_images_dir']
            # collect val files similarly
            val_files = []
            for p in patterns:
                val_files.extend(glob.glob(os.path.join(val_images_dir, p)))
            val_files = sorted(list(set(val_files)))
            val_list = [os.path.basename(x) for x in val_files]
        else:
            # fallback: use remaining files after train as val (or use small subset)
            remaining = [f for f in all_files if f not in train_list]
            # take first 20 of remaining as val (or all if small)
            val_count = cfg.get('val_count', 20)
            val_list = remaining[:val_count]

    # Build datasets and loaders
    train_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=train_list)
    val_ds = ISICDataset(cfg['images_dir'], cfg['masks_dir'], file_list=val_list)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers',0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    base_ch = cfg.get('base_ch', 16)
    model = UNet(in_channels=cfg['in_channels'], n_classes=cfg['num_classes'], base_ch=base_ch).to(device)
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

    print(f"Training samples: {len(train_list)}; Validation samples: {len(val_list)}")
    best_dice = 0.0
    for epoch in range(cfg['num_epochs']):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_mean, val_std = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_dice={val_mean:.4f}±{val_std:.4f}")
        if val_mean > best_dice:
            best_dice = val_mean
            torch.save(model.state_dict(), os.path.join(out_dir, 'unet_best.pth'))
    print('Training finished. Best val dice:', best_dice)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--train_limit', type=int, default=None, help='If set, limit training set to first N samples read from images_dir')
    parser.add_argument('--train_start', type=int, default=None, help='If set, start index (0-based) in images_dir listing')
    args = parser.parse_args()
    main(args.config, train_limit=args.train_limit, train_start=args.train_start)
