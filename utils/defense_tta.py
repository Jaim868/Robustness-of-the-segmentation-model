# utils/defense_tta.py
"""
TTA + input-denoise utilities for segmentation inference.
- Provides image <-> tensor helpers.
- Provides simple denoisers: median, gaussian, bit-depth reduce.
- Provides robust tta_predict which maps logits back to original coordinates
  before soft-averaging (fixes the common spatial-misalignment bug).
- Provides vote aggregation and consistency score.
Expectations:
- input tensors in functions are 1xCxHxW float in [0,1] (torch.Tensor).
- model returns logits shape (1,C,H,W).
"""

from PIL import Image, ImageFilter
import numpy as np
import torch
import io
from typing import List, Dict

# -----------------------
# Helpers: conversion
# -----------------------
def pil_from_tensor(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert 1xCxHxW tensor (float 0..1) -> PIL RGB image
    """
    if isinstance(img_tensor, torch.Tensor):
        arr = img_tensor.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    else:
        arr = np.array(img_tensor)
    arr = (np.clip(arr,0,1) * 255.0).astype('uint8')
    return Image.fromarray(arr)

def tensor_from_pil(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img).astype('float32') / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr,arr,arr], axis=-1)
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
    return t

# -----------------------
# Basic denoising / preprocessing
# -----------------------
def median_denoise_tensor(img_tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pil = pil_from_tensor(img_tensor)
    pil = pil.filter(ImageFilter.MedianFilter(size=kernel_size))
    return tensor_from_pil(pil)

def gaussian_denoise_tensor(img_tensor: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    pil = pil_from_tensor(img_tensor)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return tensor_from_pil(pil)

def bit_depth_reduce_tensor(img_tensor: torch.Tensor, bits: int = 6) -> torch.Tensor:
    if bits >= 8:
        return img_tensor
    levels = 2 ** bits
    t = torch.clamp(img_tensor, 0.0, 1.0)
    t_q = torch.round(t * (levels - 1)) / (levels - 1)
    return t_q

# -----------------------
# Transform specs helpers
# -----------------------
def make_transform_specs(n_transforms: int) -> List[Dict]:
    """
    Returns a list of transform specifications for tta_predict.
    Each spec is dict: {'name': 'identity'|'flip_h'|'flip_v'|'shift', 'params': {...}}
    Default ordering: identity, flip_h (if >=2), shift(dx=2,dy=0) (if >=3)
    """
    specs = [{'name':'identity'}]
    if n_transforms >= 2:
        specs.append({'name':'flip_h'})
    if n_transforms >= 3:
        specs.append({'name':'shift', 'params':{'dx':2, 'dy':0}})
    # you can extend with flip_v or more shifts if desired
    return specs

# -----------------------
# Inverse transforms for logits
# -----------------------
def _inverse_logits_by_transform(logits: torch.Tensor, transform_spec: Dict) -> torch.Tensor:
    """
    logits: tensor (1,C,H,W)
    transform_spec: {'name':..., 'params':...}
    Returns logits spatially mapped back to original orientation.
    """
    name = transform_spec.get('name','identity')
    params = transform_spec.get('params', {})
    # flip horizontal
    if name == 'flip_h':
        return logits.flip(-1)
    if name == 'flip_v':
        return logits.flip(-2)
    if name == 'shift':
        dx = params.get('dx', 0)
        dy = params.get('dy', 0)
        # inverse shift corresponds to roll by (-dy, -dx) along H,W
        return torch.roll(logits, shifts=(-dy, -dx), dims=(-2, -1))
    # identity or unknown
    return logits

# -----------------------
# Aggregation utilities
# -----------------------
def vote_aggregate(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    preds: list of tensors (1,H,W) (int labels)
    returns: aggregated label tensor (H,W) torch.LongTensor
    """
    arrs = [p.squeeze(0).cpu().numpy().astype('int64') for p in preds]
    stacked = np.stack(arrs, axis=0)  # (T,H,W)
    T,H,W = stacked.shape
    out = np.zeros((H,W), dtype='int64')
    for i in range(H):
        for j in range(W):
            vals = stacked[:,i,j]
            binc = np.bincount(vals)
            out[i,j] = int(np.argmax(binc))
    return torch.from_numpy(out).long()

def soft_average_aggregate_logits(logit_list: List[torch.Tensor]) -> torch.Tensor:
    """
    logit_list: list of tensors (1,C,H,W) (already mapped to original orientation)
    returns: argmax over average logits -> (H,W) torch.LongTensor
    """
    device = logit_list[0].device
    sum_logits = torch.zeros_like(logit_list[0], device=device)
    for lg in logit_list:
        sum_logits = sum_logits + lg.to(device)
    avg = sum_logits / float(len(logit_list))
    probs = torch.softmax(avg, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze(0).cpu().long()
    return pred

def consistency_score(preds: List[torch.Tensor]) -> float:
    """
    Fraction of pixels where all preds agree with majority (i.e. strict consistency).
    preds: list of (1,H,W) tensors
    """
    arrs = [p.squeeze(0).cpu().numpy().astype('int64') for p in preds]
    stacked = np.stack(arrs, axis=0)  # (T,H,W)
    T,H,W = stacked.shape
    same_count = 0
    total = H*W
    for i in range(H):
        for j in range(W):
            vals = stacked[:,i,j]
            binc = np.bincount(vals)
            maj = int(np.argmax(binc))
            # count pixel only if all preds equal to majority
            same_count += int((vals == maj).sum() == T)
    return float(same_count) / float(total)

# -----------------------
# Main TTA predict: robust implementation
# -----------------------
def tta_predict(model, img_tensor: torch.Tensor, device: torch.device,
                transform_specs: List[Dict], denoise_funcs: List = None,
                aggregate: str = 'soft') -> (torch.Tensor, float):
    """
    model: segmentation model (returns logits shaped (1,C,H,W))
    img_tensor: 1xCxHxW tensor in [0,1] (torch.Tensor) - can be on CPU or device
    device: torch.device or string
    transform_specs: list of transform_spec dicts (see make_transform_specs)
    denoise_funcs: list of functions tensor->tensor applied after transform (if None: identity)
    aggregate: 'soft' or 'vote'
    Returns:
      aggregated_pred (H,W) torch.LongTensor, consistency_score float
    """

    model.eval()
    if denoise_funcs is None:
        denoise_funcs = [lambda x: x]

    device = device if isinstance(device, torch.device) else torch.device(device)
    logits_mapped = []   # list of tensors (1,C,H,W) mapped back to original coordinates (CPU or same device)
    preds_hard = []      # list of (1,H,W) tensors (cpu)

    # ensure input is torch tensor on CPU for PIL conversion convenience
    img_cpu = img_tensor.detach().cpu()

    for spec in transform_specs:
        name = spec.get('name','identity')
        params = spec.get('params', {})

        # apply transform using PIL to preserve behavior
        pil = pil_from_tensor(img_cpu)
        if name == 'identity':
            pil_t = pil
        elif name == 'flip_h':
            pil_t = pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif name == 'flip_v':
            pil_t = pil.transpose(Image.FLIP_TOP_BOTTOM)
        elif name == 'shift':
            dx = params.get('dx',0); dy = params.get('dy',0)
            pil_t = pil.transform(pil.size, Image.AFFINE, (1,0,dx,0,1,dy), resample=Image.BILINEAR, fillcolor=(0,0,0))
        else:
            pil_t = pil  # fallback

        t_in = tensor_from_pil(pil_t).to(device)  # 1xCxHxW on device

        for den in denoise_funcs:
            t_den = den(t_in)
            with torch.no_grad():
                logits = model(t_den.to(device))  # (1,C,H,W) on device
            # inverse logits to original coordinate system
            inv_logits = _inverse_logits_by_transform(logits, spec)
            # move to CPU (to avoid device mixing) but keep dtype float32
            logits_mapped.append(inv_logits.detach().cpu())
            pred = torch.argmax(torch.softmax(inv_logits, dim=1), dim=1).cpu()
            preds_hard.append(pred)

    if aggregate == 'soft' and len(logits_mapped) > 0:
        # average logits (they are all in original coords)
        agg = soft_average_aggregate_logits([lg for lg in logits_mapped])
        cons = consistency_score(preds_hard)
        return agg.long(), cons
    else:
        agg = vote_aggregate(preds_hard)
        cons = consistency_score(preds_hard)
        return agg.squeeze(0).long(), cons

# 兼容旧代码的小封装（返回 transform_spec dict）
def _make_identity_transform() -> Dict:
    """返回 identity transform spec"""
    return {'name': 'identity'}

def _make_flip_transform(direction: str = 'h') -> Dict:
    """返回水平或垂直翻转的 transform spec。direction: 'h' 或 'v'。"""
    if direction == 'h':
        return {'name': 'flip_h'}
    elif direction == 'v':
        return {'name': 'flip_v'}
    else:
        raise ValueError("direction must be 'h' or 'v'")

def _make_shift_transform(dx: int = 2, dy: int = 0) -> Dict:
    """返回平移的 transform spec，params 包含 dx, dy"""
    return {'name': 'shift', 'params': {'dx': int(dx), 'dy': int(dy)}}