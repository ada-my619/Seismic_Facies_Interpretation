import torch

def dice_loss_weighted(logits, target, class_w=None, eps=1e-6):
    # logits -> probs
    probs = torch.softmax(logits, dim=1)                       # (B,C,H,W)
    C = probs.shape[1]
    target_1h = torch.nn.functional.one_hot(target, C).permute(0,3,1,2).float()

    dims = (0, 2, 3)
    inter = (probs * target_1h).sum(dims)
    denom = probs.sum(dims) + target_1h.sum(dims)

    dice_per_class = (2 * inter + eps) / (denom + eps)         # (C,)
    loss_per_class = 1 - dice_per_class

    if class_w is not None:
        class_w = class_w.to(logits.device).float()
        class_w = class_w / (class_w.mean() + 1e-12)
        class_w = torch.clamp(class_w, 0.7, 2.0)
        return (class_w * loss_per_class).sum() / class_w.sum()

    return loss_per_class.mean()

def focal_loss_from_logits(logits, target, gamma=2.0, alpha=None):
    # logits: (B,C,H,W), target: (B,H,W)
    logp = torch.nn.functional.log_softmax(logits, dim=1)  # (B,C,H,W)
    p = torch.exp(logp)
    # pick p_t for true classes
    target = target.unsqueeze(1)  # (B,1,H,W)
    logp_t = torch.gather(logp, 1, target).squeeze(1)  # (B,H,W)
    p_t = torch.gather(p, 1, target).squeeze(1)        # (B,H,W)

    loss = -((1 - p_t) ** gamma) * logp_t  # (B,H,W)

    if alpha is not None:
        # alpha: (C,) tensor of class weights
        a_t = alpha[target.squeeze(1)]     # (B,H,W)
        loss = a_t * loss

    return loss.mean()