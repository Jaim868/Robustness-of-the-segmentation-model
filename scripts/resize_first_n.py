# scripts/resize_first_n.py
import os
from PIL import Image
import argparse

def resize_first_n_images(img_dir, mask_dir, out_img_dir, out_mask_dir, n=100, size=(512,384)):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 只处理以 ISIC_ 开头的图片，按文件名排序
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png')) and f.startswith('ISIC_')])
    imgs = imgs[:n]

    print(f"Resizing {len(imgs)} images from {img_dir} -> {out_img_dir}")
    for fn in imgs:
        img_path = os.path.join(img_dir, fn)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to open image {img_path}: {e}")
            continue
        img_resized = img.resize(size, Image.BILINEAR)
        img_resized.save(os.path.join(out_img_dir, fn))

        # mask 名称规则： ISIC_0000000_segmentation.png
        base = os.path.splitext(fn)[0]
        mask_name = base + "_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')
                mask_resized = mask.resize(size, Image.NEAREST)
                mask_resized.save(os.path.join(out_mask_dir, mask_name))
            except Exception as e:
                print(f"Failed to process mask {mask_path}: {e}")
        else:
            print(f"Warning: mask not found for {fn} (expected {mask_name})")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='dataset/ISIC2018_Task1-2_Training_Input')
    parser.add_argument('--masks_dir', default='dataset/ISIC2018_Task1_Training_GroundTruth')
    parser.add_argument('--out_images', default='dataset/resized_100/images')
    parser.add_argument('--out_masks', default='dataset/resized_100/masks')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=384)
    args = parser.parse_args()

    resize_first_n_images(args.images_dir, args.masks_dir, args.out_images, args.out_masks, n=args.n, size=(args.width, args.height))
