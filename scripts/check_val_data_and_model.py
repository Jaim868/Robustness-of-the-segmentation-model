# scripts/check_val_data_and_model.py
import os, sys, json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from models.unet import UNet

cfg_path = sys.argv[1] if len(sys.argv)>1 else 'configs/config_isic.json'
ckpt = sys.argv[2] if len(sys.argv)>2 else None
out_dir = sys.argv[3] if len(sys.argv)>3 else 'outputs_check'
os.makedirs(out_dir, exist_ok=True)

with open(cfg_path) as f:
    cfg = json.load(f)
images_dir = cfg['images_dir']; masks_dir = cfg['masks_dir']
resize = cfg.get('resize', None)

# build val list: last val_count files
all_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png')) and f.startswith('ISIC_')])
val_count = cfg.get('val_count', 20)
val_list = all_files[-val_count:]

print("Val samples:", val_list[:5], "... total", len(val_list))
# check masks
for fname in val_list:
    base = os.path.splitext(fname)[0]
    mask_candidates = [os.path.join(masks_dir, base + '_segmentation.png'), os.path.join(masks_dir, base + '.png'), os.path.join(masks_dir, base + '_mask.png')]
    mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
    if mask_path is None:
        print(fname, "MASK MISSING")
        continue
    m = Image.open(mask_path).convert('L')
    if resize:
        m = m.resize((resize[0], resize[1]), resample=Image.NEAREST)
    arr = np.array(m)
    print(fname, "mask unique:", np.unique(arr), "sum:", arr.sum())

# if checkpoint provided, run one forward on first val sample
if ckpt:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_model = cfg
    model = UNet(in_channels=cfg.get('in_channels',3), n_classes=cfg.get('num_classes',2), base_ch=cfg.get('base_ch',16)).to(device)
    state = torch.load(ckpt, map_location=device)
    # find state_dict
    sd = state.get('state_dict', state) if isinstance(state, dict) else state
    new = {}
    for k,v in sd.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new[nk] = v
    model.load_state_dict(new)
    model.eval()
    # pick first sample with mask
    for fname in val_list:
        img_path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_candidates = [os.path.join(masks_dir, base + '_segmentation.png'), os.path.join(masks_dir, base + '.png')]
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
        if not mask_path:
            continue
        img = Image.open(img_path).convert('RGB')
        if resize:
            img = img.resize((resize[0], resize[1]))
        img_np = np.array(img).astype('float32')/255.0
        t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(t)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long()
            else:
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
        print("Model output logits shape:", logits.shape, "probs range:", probs.min().item(), probs.max().item())
        print("Pred sum:", pred.sum().item())
        # save pred visual
        pred_np = pred.squeeze(0).cpu().numpy().astype('uint8')*255
        Image.fromarray(pred_np).save(os.path.join(out_dir, f'{base}_pred_debug.png'))
        # save probs max
        pm = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
        import matplotlib.pyplot as plt
        plt.imshow(pm); plt.colorbar(); plt.title('max prob'); plt.savefig(os.path.join(out_dir, f'{base}_probmax.png')); plt.close()
        break
    print("Saved debug visuals to", out_dir)
else:
    print("No checkpoint provided; mask-only check done.")
