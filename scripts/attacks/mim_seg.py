# scripts/attacks/mim_seg.py
import torch

def mim_attack_on_segmentation(model, images, labels, epsilon, alpha, iters, loss_fn, device, decay=1.0):
    """
    Momentum Iterative Method for segmentation (per-batch)
    decay: momentum decay factor (typical 1.0)
    returns adv_images
    """
    images = images.to(device)
    labels = labels.to(device)
    adv = images.clone().detach()
    adv.requires_grad = True
    g = torch.zeros_like(adv).to(device)  # momentum term

    for i in range(iters):
        logits = model(adv)
        loss = loss_fn(logits, labels)
        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()
        grad = adv.grad.data
        # normalize gradient (L1 norm across batch)
        grad_norm = torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad / (grad_norm + 1e-12)
        g = decay * g + grad
        adv = adv + alpha * torch.sign(g)
        # project
        adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv.requires_grad = True
    return adv.detach()
