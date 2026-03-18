# scripts/attacks/fgsm_seg.py
import torch

def fgsm_attack_on_segmentation(model, images, labels, epsilon, loss_fn, device):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    logits = model(images)
    loss = loss_fn(logits, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.data.sign()
    adv = images + epsilon * grad_sign
    adv = torch.clamp(adv, 0.0, 1.0).detach()
    return adv
