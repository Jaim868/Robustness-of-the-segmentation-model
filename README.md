# Evaluating Adversarial Robustness in Medical Image Segmentation 🛡️🩺

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Project](https://img.shields.io/badge/Project-Honours_Degree-blue.svg)]()

This repository contains the official implementation for the Honours Project: **Evaluating Adversarial Robustness in Medical Image Segmentation**. 

It provides a comprehensive, modular PyTorch framework to train medical image segmentation models (U-Net), evaluate their vulnerability against gradient-based adversarial attacks (FGSM, PGD, BIM, MIM), and harden them using advanced optimization techniques including **Spectral Normalization (SN)** and **TRADES** (Tradeoff-inspired Adversarial Defense via Surrogate-loss).

---

## 📑 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Environment Setup](#-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
-[End-to-End Usage Workflow](#-end-to-end-usage-workflow)
  -[Step 1: Train the Baseline Model](#step-1-train-the-baseline-model)
  -[Step 2: Vulnerability Assessment (Attacks)](#step-2-vulnerability-assessment-attacks)
  -[Step 3: Adversarial Training (Trade-off Analysis)](#step-3-adversarial-training-trade-off-analysis)
  -[Step 4: Advanced Defense (TRADES + SN)](#step-4-advanced-defense-trades--spectral-normalization)
  -[Step 5: Qualitative Visualization](#step-5-qualitative-visualization)
- [Results Summary](#-results-summary)

---

## ✨ Features
- **Lightweight Segmentation:** Parameterized U-Net architecture (`base_ch=8, 16, 32`) optimized for consumer-grade GPUs (8GB VRAM).
- **Comprehensive Attack Suite:** Implementations of $L_\infty$-bounded white-box attacks specifically adapted for dense prediction tasks.
- **Advanced Defenses:** Linearly Weighted Adversarial Training, Spectral Normalization, and TRADES.
- **Robust Evaluation Pipeline:** Automated calculation of Clean/Robust Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).

---

## 📂 Project Structure

```text
├── configs/
│   └── config_isic.json         # Global hyperparameters and path configurations
├── dataset/                     # Place your images and masks here
├── models/
│   └── unet.py                  # U-Net architecture with Spectral Normalization support
├── scripts/
│   ├── attacks/                 # Attack algorithms (fgsm_seg.py, pgd_seg.py, etc.)
│   ├── train_unet.py            # Standard ERM training script
│   ├── train_tradeoff.py        # Adversarial training script with alpha weighting
│   ├── train_sn.py              # Adversarial training + Spectral Normalization
│   ├── train_trades.py          # TRADES optimization script
│   ├── eval_adv.py              # Quantitative evaluation script
│   └── visualize_comparison.py  # Script for generating side-by-side visual comparisons
├── utils/
│   ├── dataset.py               # PyTorch Dataset loader and augmentations
│   └── metrics.py               # Numpy-based Dice and IoU calculations
└── README.md                    # This document
