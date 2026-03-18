# scripts/generate_splits.py
import os, json, random

def make_splits(images_dir, out_path, train_frac=0.7, val_frac=0.15, seed=42):
    fnames = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    random.seed(seed)
    random.shuffle(fnames)
    n = len(fnames)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = {
        'train': fnames[:n_train],
        'val': fnames[n_train:n_train+n_val],
        'test': fnames[n_train+n_val:]
    }
    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {out_path} (total {n} images)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--out', default='splits.json')
    args = parser.parse_args()
    make_splits(args.images_dir, args.out)
