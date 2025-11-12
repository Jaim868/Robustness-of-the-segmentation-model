ENVIRONMENT
conda activate adv_seg

EDV:
python scripts\eval_adv.py --config configs/config_isic.json --ckpt outputs/unet_best.pth --attack fgsm --eps 0.0157 --attack_iters 10 --val_start_name ISIC_0000101.jpg --val_count 20 --defense tta --defense_bits 6 --tta_transforms 3

ADV_TRAINED:
python scripts\adversarial_training.py --config configs/config_isic.json --out outputs
