# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils


# --- 辅助函数：应用谱归一化 ---
def apply_sn(layer, use_sn):
    if use_sn:
        return utils.spectral_norm(layer)
    return layer


# --- NonLocalBlock (保持不变，但为了兼容性可以加个参数，这里暂时不加SN以保持变量控制) ---
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        self.sub_sample = sub_sample
        if sub_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(self.max_pool(x)).view(batch_size, self.inter_channels, -1)
        else:
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        return W_y + x


# --- 修改后的 ConvBlock ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, spectral_norm=False):
        super().__init__()
        # 如果启用 SN，包裹 Conv2d
        self.double_conv = nn.Sequential(
            apply_sn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), spectral_norm),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            apply_sn(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), spectral_norm),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# --- 修改后的 Down ---
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, spectral_norm=False):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch, spectral_norm=spectral_norm)
        )

    def forward(self, x):
        return self.pool_conv(x)


# --- 修改后的 Up ---
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, spectral_norm=False):
        super().__init__()
        # 上采样层通常也建议加 SN，或者保持原样。这里我们也加上 SN。
        self.up = apply_sn(nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2), spectral_norm)
        self.conv = ConvBlock(in_ch, out_ch, spectral_norm=spectral_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- 修改后的 OutConv ---
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch, spectral_norm=False):
        super().__init__()
        self.conv = apply_sn(nn.Conv2d(in_ch, out_ch, kernel_size=1), spectral_norm)

    def forward(self, x):
        return self.conv(x)


# --- 修改后的 UNet 主类 ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_ch=32, denoise=False, spectral_norm=False):
        super().__init__()
        self.denoise = denoise
        self.spectral_norm = spectral_norm

        self.inc = ConvBlock(in_channels, base_ch, spectral_norm)
        self.down1 = Down(base_ch, base_ch * 2, spectral_norm)
        self.down2 = Down(base_ch * 2, base_ch * 4, spectral_norm)
        self.down3 = Down(base_ch * 4, base_ch * 8, spectral_norm)

        if self.denoise:
            self.dn2 = NonLocalBlock(base_ch * 4)
            self.dn3 = NonLocalBlock(base_ch * 8)

        self.up1 = Up(base_ch * 8, base_ch * 4, spectral_norm)
        self.up2 = Up(base_ch * 4, base_ch * 2, spectral_norm)
        self.up3 = Up(base_ch * 2, base_ch, spectral_norm)
        self.outc = OutConv(base_ch, n_classes, spectral_norm)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.denoise:
            x3 = self.dn2(x3)
            x4 = self.dn3(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits