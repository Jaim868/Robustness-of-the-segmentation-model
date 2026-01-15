import torch
import torch.nn as nn
import torch.nn.functional as F


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                distance='l_inf'):
    """
    Fixed TRADES implementation for Segmentation to prevent model collapse.
    """
    # 1. 设置模型模式
    model.eval()

    batch_size = len(x_natural)
    # 获取像素总数 (B * H * W)，用于归一化
    num_pixels = x_natural.size(0) * x_natural.size(2) * x_natural.size(3)

    # --- Generate Adversarial Example ---
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                logits_adv = model(x_adv)
                logits_clean = model(x_natural)

                # [Fix 1] Inner Loop KL: 使用 sum 并除以像素数
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean, dim=1),
                    reduction='sum'
                )
                loss_kl = loss_kl / num_pixels  # 严格归一化

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    # --- Calculate Final Loss ---
    x_adv = x_adv.detach()

    logits_clean = model(x_natural)
    logits_adv = model(x_adv)

    # 1. Clean Accuracy Loss (Cross Entropy)
    criterion_ce = nn.CrossEntropyLoss()
    loss_natural = criterion_ce(logits_clean, y)

    # 2. Robustness Loss (KL Divergence)
    # [Fix 2] Outer Loop KL: 同样使用 sum 并除以像素数，保持与 Clean Loss 量级一致
    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_clean, dim=1),
        reduction='sum'
    )
    loss_robust = loss_robust / num_pixels

    # [Fix 3] 动态调整：如果 KL 仍然过大，这里可以不乘 Beta，或者乘一个更小的数
    # 但经过上述归一化后，loss_robust 应该在 0.0x 到 0.x 之间，与 loss_natural (0.x) 相当
    loss_total = loss_natural + beta * loss_robust

    return loss_total, loss_natural.item(), loss_robust.item()