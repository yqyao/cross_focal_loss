import torch
from torch.nn import functional as F


def cross_sigmoid_focal_loss(inputs,
                             targets,
                             gt_avoid_classes=None,
                             alpha=0.25,
                             gamma=2,
                             ignore_label=-1,
                             reduction="sum"):
    """
    Arguments:
       - inputs: inputs Tensor (N * C)
       - targets: targets Tensor (N)
       - gt_avoid_classes: [set(), set()...] neg label need to be avoided for each class
       - alpha: focal loss alpha
       - gamma: focal loss gamma
       - ignore_label: ignore_label default = -1
       - reduction: default = sum
    """

    def get_classes_idx(gt_avoid_classes, neg_target):
        classes_idx = []
        for idx, item in enumerate(gt_avoid_classes):
            if neg_target in item:
                classes_idx.append(idx)
        return classes_idx

    assert gt_avoid_classes is not None, "gt_avoid_classes must be provided"
    sample_mask = (targets != ignore_label)
    if (~sample_mask).sum() > 0:
        inputs = inputs[sample_mask]
        targets = targets[sample_mask]

    cross_mask = inputs.new_ones(inputs.shape, dtype=torch.int8)
    t_mask = inputs.new_zeros(inputs.shape[0], dtype=torch.int8)

    neg_targets = set()
    for item in gt_avoid_classes:
        neg_targets = neg_targets.union(item)

    for neg_target in neg_targets:
        neg_mask = targets == neg_target
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
        if len(neg_idx) > 0:
            t_mask |= neg_mask
            cls_neg_idx = get_classes_idx(gt_avoid_classes, neg_target)
            cross_mask[neg_idx, cls_neg_idx] = 0
    vaild_idx = torch.nonzero(1 - t_mask, as_tuple=False).reshape(-1)
    pos_num = max(vaild_idx.shape[0], 1)
    expand_label_targets = torch.zeros_like(inputs)
    expand_label_targets[vaild_idx, targets[vaild_idx] - 1] = 1

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 expand_label_targets,
                                                 reduction="none")
    p_t = p * expand_label_targets + (1 - p) * (1 - expand_label_targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * expand_label_targets + (1 - alpha) * (
            1 - expand_label_targets)
        loss = alpha_t * loss

    loss *= cross_mask

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
        loss /= pos_num
    return loss


if __name__ == '__main__':
    inputs = torch.rand(10, 2)
    targets = torch.Tensor([1, 1, 1, 3, 3, 3, 4, 4, 4, 2]).long()
    gt_avoid_classes = [{4}, {3}]
    loss = cross_sigmoid_focal_loss(inputs, targets, gt_avoid_classes, reduction="sum")
