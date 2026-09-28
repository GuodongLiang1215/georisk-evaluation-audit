import torch
import torch.nn.functional as F


def soft_dice_loss(logits, target, ignore_index=255):
    num_classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probabilities = torch.softmax(logits, dim=1) * valid
    one_hot = one_hot * valid
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice.mean()


def segmentation_loss(logits, target, ignore_index=255):
    if not torch.any(target != ignore_index):
        return logits.sum() * 0.0
    cross_entropy = F.cross_entropy(logits, target, ignore_index=ignore_index)
    dice = soft_dice_loss(logits, target, ignore_index)
    return 0.5 * cross_entropy + 0.5 * dice
