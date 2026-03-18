import torch


def pgd_attack_on_segmentation(model, images, labels, eps, alpha, iters, loss_fn, device):
    """
    正确的PGD攻击实现
    """
    ori_images = images.data

    # 初始化扰动
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)

    for i in range(iters):
        delta.requires_grad = True
        inputs = ori_images + delta
        outputs = model(inputs)

        # 使用与训练相同的损失函数
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        # 更新扰动
        delta_grad = delta.grad.detach()
        delta = delta + alpha * delta_grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(ori_images + delta, 0, 1) - ori_images

        delta = delta.detach()

    return ori_images + delta