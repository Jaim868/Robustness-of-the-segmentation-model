# scripts/fake_data_generator.py
# Generate small fake dataset for testing pipeline
import os
from PIL import Image, ImageDraw
import random

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./dataset')
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'masks'), exist_ok=True)
    for i in range(args.n):
        img = Image.new('RGB', (args.size, args.size), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1,3)):
            x0 = random.randint(0, args.size//2)
            y0 = random.randint(0, args.size//2)
            x1 = random.randint(args.size//2, args.size-1)
            y1 = random.randint(args.size//2, args.size-1)
            draw.ellipse([x0,y0,x1,y1], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img_name = f'img_{i:04d}.jpg'
        img.save(os.path.join(args.out,'images',img_name))
        mask = Image.new('L', (args.size, args.size), 0)
        drawm = ImageDraw.Draw(mask)
        drawm.ellipse([args.size//4, args.size//4, args.size*3//4, args.size*3//4], fill=255)
        mask.save(os.path.join(args.out,'masks',img_name.replace('.jpg','_mask.png')))
    print('Fake dataset generated at', args.out)
