# scripts/preprocess.py
# Minimal preprocess for ISIC-style dataset: resize images and masks.
import os
from PIL import Image

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--masks_dir', required=True)
    parser.add_argument('--out_dir', default='./dataset')
    parser.add_argument('--width', type=int, default=1022)
    parser.add_argument('--height', type=int, default=767)
    args = parser.parse_args()

    out_images = os.path.join(args.out_dir, 'ISIC2018_Task1-2_Training_Input')
    out_masks = os.path.join(args.out_dir, 'ISIC2018_Task1_Training_GroundTruth')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    imgs = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in imgs:
        try:
            img = Image.open(os.path.join(args.images_dir, f)).convert('RGB')
            img = img.resize((args.width, args.height))
            img.save(os.path.join(out_images, f))
        except Exception as e:
            print(f"Failed to process image {f}: {e}")

        base = os.path.splitext(f)[0]
        # masks naming in original is often base + '_segmentation.png'
        mask_candidates = [
            base + '_segmentation.png',
            base + '_segmentation.jpg',
            base + '.png',
            base + '_mask.png'
        ]
        found = False
        for mname in mask_candidates:
            mpath = os.path.join(args.masks_dir, mname)
            if os.path.exists(mpath):
                try:
                    mask = Image.open(mpath).convert('L')
                    mask = mask.resize((args.width, args.height))
                    mask.save(os.path.join(out_masks, base + '_segmentation.png'))
                    found = True
                    break
                except Exception as e:
                    print(f"Failed to process mask {mpath}: {e}")
        if not found:
            print(f"Warning: mask not found for image {f}; looked for {mask_candidates}")
    print('Preprocess done. Resized images saved to', out_images)
    print('Resized masks saved to', out_masks)
