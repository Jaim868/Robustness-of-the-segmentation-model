# scripts/visualize_case.py
import os, json, argparse
from PIL import Image
import numpy as np
import torch
from models.unet import UNet
from utils.dataset import ISICDataset
from scripts.attacks.fgsm_seg import fgsm_attack_on_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_isic.json')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--attack', choices=['none','fgsm','pgd'], default='fgsm')
    parser.add_argument('--eps', type=float, default=4/255)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    device = torch.device(cfg.get('device') if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, n_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    img_path = os.path.join(cfg['images_dir'], args.case)
    mask_path = os.path.join(cfg['masks_dir'], args.case.replace('.jpg','_mask.png'))
    img = Image.open(img_path).convert('RGB').resize((256,256))
    mask = Image.open(mask_path).convert('L').resize((256,256))
    img_np = np.array(img).astype('float32')/255.0
    mask_np = (np.array(mask)>127).astype('uint8')
    img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()

    if args.attack == 'none':
        adv = img_t
    else:
        adv = fgsm_attack_on_segmentation(model, img_t, torch.from_numpy(mask_np).unsqueeze(0), args.eps, lambda x,y: torch.nn.CrossEntropyLoss()(x,y.to(device)), device)

    with torch.no_grad():
        logits_clean = model(img_t.to(device))
        pred_clean = torch.argmax(torch.softmax(logits_clean, dim=1), dim=1).cpu().numpy()[0]
        logits_adv = model(adv.to(device))
        pred_adv = torch.argmax(torch.softmax(logits_adv, dim=1), dim=1).cpu().numpy()[0]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,4, figsize=(12,4))
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[1].imshow((adv.squeeze(0).permute(1,2,0).cpu().numpy()-img_np)*10 + 0.5)
    axes[1].set_title('Perturbation x10')
    axes[2].imshow(img_np)
    axes[2].contour(pred_clean, levels=[0.5], colors='r')
    axes[2].set_title('Clean Pred')
    axes[3].imshow(img_np)
    axes[3].contour(pred_adv, levels=[0.5], colors='r')
    axes[3].set_title('Adv Pred')
    for ax in axes:
        ax.axis('off')
    outp = os.path.join(cfg.get('out_dir','outputs'), f"vis_{args.case.replace('.jpg','')}_eps{args.eps:.4f}.png")
    plt.savefig(outp, bbox_inches='tight')
    print('Saved', outp)
