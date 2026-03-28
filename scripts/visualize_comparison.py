# scripts/visualize_comparison.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复 OpenMP 冲突

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import glob

# 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation


# --- 1. 简单的模型加载函数 ---
def load_model(ckpt_path, device, base_ch=32, num_classes=2):
    # 初始化模型结构
    model = UNet(in_channels=3, n_classes=num_classes, base_ch=base_ch).to(device)

    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return model

    print(f"Loading weights from: {ckpt_path}")
    try:
        # PyTorch 2.6+ 安全加载
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版本 PyTorch
        state = torch.load(ckpt_path, map_location=device)

    # 处理字典嵌套
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


# --- 2. 纯净的叠加函数 ---
def overlay_mask(img_np, mask_np, color, alpha=0.4):
    """
    img_np: (H, W, 3) float [0, 1]
    mask_np: (H, W) int 0 or 1
    color: list [r, g, b] e.g. [1, 0, 0] for red
    """
    # 复制一份，不修改原图
    out = img_np.copy()

    # 找到 mask 为 1 的区域
    mask_indices = mask_np > 0

    # 简单的 Alpha Blending
    for c in range(3):
        out[mask_indices, c] = out[mask_indices, c] * (1 - alpha) + color[c] * alpha

    return np.clip(out, 0, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 搜索图片 ---
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    all_imgs = []
    for ext in extensions:
        all_imgs.extend(glob.glob(os.path.join(args.img_dir, ext)))
    all_imgs = sorted(list(set(all_imgs)))

    if len(all_imgs) == 0:
        print(f"Error: No images found in {args.img_dir}")
        return

    # 确定要画哪张图
    img_path = ""
    if args.image_name:
        potential = os.path.join(args.img_dir, args.image_name)
        if os.path.exists(potential):
            img_path = potential

    if not img_path:
        idx = args.index if args.index < len(all_imgs) else 0
        img_path = all_imgs[idx]

    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    # --- 搜索 Mask ---
    possible_masks = [base_name + ".png", base_name + "_mask.png", base_name + "_segmentation.png", filename]
    mask_path = None
    for m in possible_masks:
        p = os.path.join(args.mask_dir, m)
        if os.path.exists(p):
            mask_path = p
            break

    if not mask_path:
        print(f"Error: Mask not found for {filename}")
        return

    print(f"Target Image: {filename}")

    # --- 数据预处理 ---
    # 统一 Resize 到 512x384 (或者你的模型训练尺寸)
    H, W = 384, 512

    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # 读取
    raw_img = Image.open(img_path).convert('RGB')
    raw_mask = Image.open(mask_path).convert('L')

    # 转 Tensor
    img_t = transform(raw_img).unsqueeze(0).to(device)  # (1, 3, H, W)
    mask_t = mask_transform(raw_mask).to(device)
    mask_t = (mask_t > 0.5).long()  # (1, 1, H, W) -> 保持维度用于计算Loss

    # 准备用于绘图的 Numpy 数组 (H, W, 3)
    np_img = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np_mask_gt = mask_t.squeeze().cpu().numpy()

    # --- 加载模型 ---
    # 1. Baseline
    model_base = load_model(args.ckpt_base, device, base_ch=args.base_ch)
    # 2. Adv Train
    model_adv = load_model(args.ckpt_adv, device, base_ch=args.base_ch)

    criterion = nn.CrossEntropyLoss()

    # --- 定义绘图列表 ---
    plot_list = []

    # 1. Original Image (纯原图)
    plot_list.append({
        "title": "Original Image",
        "data": np_img
    })

    # 2. Ground Truth (原图 + 绿色 Mask)
    gt_viz = overlay_mask(np_img, np_mask_gt, color=[0, 1, 0])  # Green
    plot_list.append({
        "title": "Ground Truth",
        "data": gt_viz
    })

    # 3. Baseline (Under Attack)
    print("Attacking Baseline...")
    # 生成攻击样本
    adv_img_base = pgd_attack_on_segmentation(
        model_base, img_t, mask_t,
        eps=16 / 255, alpha=1 / 255, iters=10,
        loss_fn=criterion, device=device
    )
    # 预测
    with torch.no_grad():
        out_base = model_base(adv_img_base)
        pred_base = torch.argmax(torch.softmax(out_base, dim=1), dim=1).squeeze().cpu().numpy()

    # 叠加红色 Mask
    base_viz = overlay_mask(np_img, pred_base, color=[1, 0, 0])  # Red
    plot_list.append({
        "title": "Baseline\n(Under Attack)",
        "data": base_viz
    })

    # 4. Adv-Train (Under Attack)
    print("Attacking Adv-Model...")
    # 生成攻击样本 (针对 Adv 模型生成攻击)
    adv_img_adv = pgd_attack_on_segmentation(
        model_adv, img_t, mask_t,
        eps=24 / 255, alpha=1 / 255, iters=10,
        loss_fn=criterion, device=device
    )
    # 预测
    with torch.no_grad():
        out_adv = model_adv(adv_img_adv)
        pred_adv = torch.argmax(torch.softmax(out_adv, dim=1), dim=1).squeeze().cpu().numpy()

    # 叠加红色 Mask
    adv_viz = overlay_mask(np_img, pred_adv, color=[1, 0, 0])  # Red
    plot_list.append({
        "title": "Adv-Train\n(Under Attack)",
        "data": adv_viz
    })

    # --- 绘图 ---
    print("Saving plot...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i, item in enumerate(plot_list):
        axs[i].imshow(item['data'])
        axs[i].set_title(item['title'], fontsize=14)
        axs[i].axis('off')  # 隐藏坐标轴

    plt.tight_layout()

    out_path = os.path.join(args.out_dir, f"vis_clean_{filename}_4.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Done! Saved to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--img_dir', required=True, help="Path to images folder")
    parser.add_argument('--mask_dir', required=True, help="Path to masks folder")
    parser.add_argument('--out_dir', default='outputs')

    # 图片选择
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--image_name', type=str, default=None)

    # 模型路径
    parser.add_argument('--ckpt_base', required=True, help="Path to Baseline model")
    parser.add_argument('--ckpt_adv', required=True, help="Path to Adversarial model")

    # 模型参数
    parser.add_argument('--base_ch', type=int, default=8, help="Base channels (e.g., 8, 16, 32)")

    args = parser.parse_args()
    main(args)