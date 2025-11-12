# utils/defenses.py
from PIL import Image, ImageFilter
import numpy as np
import io
import torch

def pil_from_tensor(img_tensor):
    # img_tensor: 1xCxHxW tensor in [0,1]
    arr = img_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    arr = (arr*255).astype('uint8')
    return Image.fromarray(arr)

def tensor_from_pil(pil_img, device=None):
    import torch
    arr = np.array(pil_img).astype('float32')/255.0
    if arr.ndim==2:
        arr = np.stack([arr,arr,arr], axis=-1)
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()
    if device is not None:
        t = t.to(device)
    return t

def median_denoise_tensor(img_tensor, kernel_size=3):
    pil = pil_from_tensor(img_tensor)
    pil = pil.filter(ImageFilter.MedianFilter(size=kernel_size))
    return tensor_from_pil(pil)

def gaussian_blur_tensor(img_tensor, radius=1):
    pil = pil_from_tensor(img_tensor)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return tensor_from_pil(pil)

def jpeg_compress_tensor(img_tensor, quality=75):
    pil = pil_from_tensor(img_tensor)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    pil2 = Image.open(buf).convert('RGB')
    return tensor_from_pil(pil2)

def bit_depth_reduce_tensor(img_tensor, bits=6):
    # reduce bit-depth per channel then rescale to [0,1]
    arr = (img_tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).astype('uint8')
    shift = 8 - bits
    arr = (arr >> shift) << shift
    import torch
    t = torch.from_numpy(arr.astype('float32')/255.0).permute(2,0,1).unsqueeze(0).float()
    return t

def apply_defense_pipeline(img_tensor, methods=None, device=None):
    """
    methods: list of tuples, e.g. [('median',3), ('jpeg',80)]
    returns tensor in same device as img_tensor (cpu/gpu)
    """
    if methods is None:
        methods = [('median',3)]
    out = img_tensor.cpu()
    for m in methods:
        name = m[0]
        arg = m[1] if len(m)>1 else None
        if name == 'median':
            out = median_denoise_tensor(out, kernel_size=arg if arg else 3)
        elif name == 'gauss':
            out = gaussian_blur_tensor(out, radius=arg if arg else 1)
        elif name == 'jpeg':
            out = jpeg_compress_tensor(out, quality=arg if arg else 80)
        elif name == 'bitreduce':
            out = bit_depth_reduce_tensor(out, bits=arg if arg else 6)
        else:
            # unknown -> skip
            pass
    if device is not None:
        return out.to(device)
    return out
