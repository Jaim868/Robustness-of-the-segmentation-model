# utils/robust_aug.py
import torch
import numpy as np


# --- 1. 高斯噪声注入 (Gaussian Noise Injection) ---
class AddGaussianNoise(object):
    """
    向 Tensor 添加高斯噪声。
    通常用于训练循环中，在输入模型之前应用。
    """

    def __init__(self, mean=0., std=0.1, p=0.5):
        """
        Args:
            mean (float): 噪声均值
            std (float): 噪声标准差 (控制噪声强度)
            p (float): 应用噪声的概率
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): 输入图像 (B, C, H, W)，通常已经归一化到 [0, 1]
        Returns:
            Tensor: 加噪后的图像
        """
        if np.random.rand() > self.p:
            return tensor

        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noisy_tensor = tensor + noise
        # 确保像素值仍然在合法范围内 (假设输入是 0-1)
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, p={self.p})'


# --- 2. Mixup 数据增强 ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''
    返回混合后的输入 (mixed_x) 以及两组标签 (y_a, y_b) 和混合比例 (lam)。

    Args:
        x (Tensor): 输入图像 batch
        y (Tensor): 输入标签/Mask batch
        alpha (float): Beta 分布参数，通常取 0.2, 0.4 或 1.0。
                       alpha 越大，混合程度越强；alpha 趋近 0 则接近原图。
    '''
    if alpha > 0:
        # 从 Beta 分布采样 lambda
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # 生成随机索引用于打乱 batch
    index = torch.randperm(batch_size).to(device)

    # 混合输入图像
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # 获取对应的两组标签（不直接混合标签，而是保留两个标签用于计算 Loss）
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Mixup 的 Loss 计算函数。
    Loss = lambda * Loss(pred, y_a) + (1 - lambda) * Loss(pred, y_b)

    Args:
        criterion (func): 原本的损失函数 (e.g., CE + Dice)
        pred (Tensor): 模型的预测输出
        y_a (Tensor): 原始标签
        y_b (Tensor): 打乱后的标签
        lam (float): 混合比例
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)