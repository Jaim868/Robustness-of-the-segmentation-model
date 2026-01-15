# scripts/visualize_comparison.py
import os

# 解决 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 确保路径正确
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation


# --- 辅助函数：加载模型 ---
def load_model(ckpt_path, device, base_ch=32, num_classes=2, sn=False):
    # 根据是否使用 Spectral Norm 初始化不同的模型结构
    model = UNet(in_channels=3, n_classes=num_classes, base_ch=base_ch, spectral_norm=sn).to(device)

    if not os.path.exists(ckpt_path):
        print(f"[Warning] Checkpoint not found: {ckpt_path}. Using Random Init Model (for placeholder).")
        return model

    try:
        # PyTorch 2.6+ 兼容性修复
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    # 兼容处理 state_dict
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']

    # 移除 module. 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


# --- 辅助函数：Mask 叠加可视化 ---
def overlay_mask(img_tensor, mask_tensor, color=(1, 0, 0), alpha=0.5):
    """
    img_tensor: (3, H, W) float, 0-1
    mask_tensor: (H, W) int, 0 or 1
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()

    # 创建彩色 Mask
    color_mask = np.zeros_like(img_np)
    for i in range(3):
        color_mask[:, :, i] = color[i]

    # 融合
    mask_indices = mask_np > 0
    # 尺寸保护
    if mask_indices.shape != img_np.shape[:2]:
        mask_indices = mask_np > 0

    img_np[mask_indices] = img_np[mask_indices] * (1 - alpha) + color_mask[mask_indices] * alpha

    return np.clip(img_np, 0, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    import glob
    all_imgs = sorted(glob.glob(os.path.join(args.img_dir, 'ISIC_*.jpg')))
    if len(all_imgs) == 0:
        print(f"Error: No images found in {args.img_dir}")
        return

    # 图片选择逻辑：优先文件名，其次索引
    if args.image_name:
        img_path = os.path.join(args.img_dir, args.image_name)
        if not os.path.exists(img_path):
            # 尝试在列表中查找
            candidates = [x for x in all_imgs if args.image_name in x]
            if candidates:
                img_path = candidates[0]
            else:
                print(f"Error: Image {args.image_name} not found.")
                return
        filename = os.path.basename(img_path)
    else:
        if args.index >= len(all_imgs):
            print(f"Error: Index {args.index} out of range.")
            return
        img_path = all_imgs[args.index]
        filename = os.path.basename(img_path)

    # 文件名处理
    base_name = os.path.splitext(filename)[0]
    mask_name = base_name + "_segmentation.png"
    mask_path = os.path.join(args.mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"Error: Mask not found at {mask_path}")
        return

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((384, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    img_t = transform(img).unsqueeze(0).to(device)
    mask_t = mask_transform(mask).to(device)

    # 保持 (1, H, W) 用于计算 Loss
    mask_t = (mask_t > 0.5).long()

    print(f"Visualizing Image: {filename}")

    # 2. 加载四个模型
    print(f"Loading models with base_ch={args.base_ch}...")

    # A. Baseline
    model_base = load_model(args.ckpt_base, device, base_ch=args.base_ch, sn=False)
    # B. Adv Train
    model_adv = load_model(args.ckpt_adv, device, base_ch=args.base_ch, sn=False)
    # C. Spectral Norm
    model_sn = load_model(args.ckpt_sn, device, base_ch=args.base_ch, sn=True)
    # D. TRADES (注意：我们在 train_trades.py 里使用了 sn=True)
    model_trades = load_model(args.ckpt_trades, device, base_ch=args.base_ch, sn=True)

    criterion = nn.CrossEntropyLoss()

    # 3. 定义对比列表
    models = [
        ('Baseline', model_base),
        (f'Adv-Train (α={args.alpha})', model_adv),
        ('Spectral Norm', model_sn),
        (f'TRADES (β={args.beta})', model_trades)
    ]

    results = []

    # (1) 原始图
    results.append({
        'title': 'Original Image',
        'pixels': img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    })
    # (2) GT
    results.append({
        'title': 'Ground Truth',
        'pixels': overlay_mask(img_t.squeeze(0), mask_t.squeeze(0), color=(0, 1, 0))  # 绿色 GT
    })

    print("Generating attacks and predictions...")
    for name, model in models:
        # 使用统一的 PGD 攻击参数进行测试，公平对比模型鲁棒性
        adv_img = pgd_attack_on_segmentation(
            model, img_t, mask_t,
            eps=4 / 255, alpha=1 / 255, iters=10,
            loss_fn=criterion, device=device
        )

        with torch.no_grad():
            logits = model(adv_img)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        viz = overlay_mask(img_t.squeeze(0), pred.squeeze(0), color=(1, 0, 0))  # 红色 预测
        results.append({'title': f'{name}\n(Under Attack)', 'pixels': viz})

    # 4. 绘图 (1行6列)
    print("Plotting...")
    fig, axs = plt.subplots(1, 6, figsize=(24, 5))  # 宽度增加以容纳6张图

    for i, res in enumerate(results):
        axs[i].imshow(res['pixels'])
        axs[i].set_title(res['title'], fontsize=11)
        axs[i].axis('off')

    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f'vis_trades_idx{args.index if not args.image_name else filename}.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='dataset/resized_100/images')  # 修改为你的图片路径
    parser.add_argument('--mask_dir', default='dataset/resized_100/masks')
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--index', type=int, default=5, help='Index of image to visualize')
    parser.add_argument('--image_name', type=str, default=None, help='Specific image filename (e.g. ISIC_0000007.jpg)')
    parser.add_argument('--alpha', type=str, default='0.7', help='Alpha value for label')
    parser.add_argument('--beta', type=str, default='0.5', help='Beta value for TRADES label')

    # 新增 base_ch 参数，默认为 8 (根据你的报错推断)
    parser.add_argument('--base_ch', type=int, default=8, help='Base channels for UNet (must match training)')

    # 模型路径 (你需要修改为你实际训练好的路径)
    parser.add_argument('--ckpt_base', default='outputs/unet_best.pth', help='Path to baseline model')
    parser.add_argument('--ckpt_adv', default='outputs/tradeoff_experiment/unet_alpha_0.70.pth',
                        help='Path to adv trained model')
    parser.add_argument('--ckpt_sn', default='outputs/sn_train/unet_sn_best.pth', help='Path to spectral norm model')
    parser.add_argument('--ckpt_trades', default='outputs/trades_fix/unet_trades_beta0.5_best.pth')

    args = parser.parse_args()
    main(args)

