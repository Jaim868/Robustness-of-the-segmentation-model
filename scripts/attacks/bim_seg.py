# scripts/attacks/bim_seg.py
import torch

def bim_attack_on_segmentation(model, images, labels, epsilon, alpha, iters, loss_fn, device):
    """
    BIM = iterative FGSM (without random start)
    images: torch tensor  (B,C,H,W)
    labels: torch tensor  (B,H,W) long
    epsilon: L_inf bound (e.g. 4/255)
    alpha: step size per iteration (e.g. epsilon/iter_steps*1.0)
    iters: number of iterations
    loss_fn: function (logits, target) -> loss scalar
    device: torch.device
    returns adv_images (same shape)
    """
    images = images.to(device)
    labels = labels.to(device)
    adv = images.clone().detach()
    adv.requires_grad = True

    for i in range(iters):
        logits = model(adv)
        loss = loss_fn(logits, labels)
        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()
        # sign gradient
        grad_sign = adv.grad.data.sign()
        adv = adv + alpha * grad_sign
        # project into L_inf ball around images
        adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
        adv = torch.clamp(adv, 0.0, 1.0).detach()
        adv.requires_grad = True
    return adv.detach()
