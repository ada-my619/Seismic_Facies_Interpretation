import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from seismic_facies.models.diff_model.utils import reverse_diffusion

def evaluate_metrics_streaming(model, data_loader, device, n_classes=6):
    model.eval()
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)

    def fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        return np.bincount(
            n_class * label_true[mask] + label_pred[mask],
            minlength=n_class**2
        ).reshape(n_class, n_class)

    with torch.no_grad():
        for batch in data_loader:
            X, y = batch[0], batch[1]      # works whether batch has (X,y) or (X,y,dir)
            X = X.to(device)
            y = y.to(device)
            if y.ndim == 4:
                y = y.squeeze(1)

            logits = model(X)
            pred = torch.argmax(logits, dim=1)

            lt = y.detach().cpu().numpy().astype(np.int64)
            lp = pred.detach().cpu().numpy().astype(np.int64)

            for t, p in zip(lt, lp):
                hist += fast_hist(t.flatten(), p.flatten(), n_classes)

    # metrics (same as runningScore)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-12)
    mean_acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-12)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum() + 1e-12)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    scores = {
        "Pixel Acc: ": float(acc),
        "Class Accuracy: ": acc_cls,
        "Mean Class Acc: ": float(mean_acc_cls),
        "Freq Weighted IoU: ": float(fwavacc),
        "Mean IoU: ": float(mean_iu),
        "confusion_matrix": hist
    }
    cls_iou = dict(zip(range(n_classes), iu))
    return scores, cls_iou

def plot_confusion_matrix(cm, class_names=None, normalize=False):
    """
    cm : ndarray (C,C)
    normalize : if True, normalize rows (GT-wise)
    """

    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    # write values inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > cm.max()/2 else "black")

    plt.tight_layout()
    plt.show()

def evaluate_metrics_diff(model, data_loader, device, alphas, alphas_bar, betas, T=1000,n_classes=6):
    model.eval()
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)

    def fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        return np.bincount(
            n_class * label_true[mask] + label_pred[mask],
            minlength=n_class**2
        ).reshape(n_class, n_class)

    with torch.no_grad():
        for batch in tqdm(data_loader):
            X, y = batch[0], batch[1]      # works whether batch has (X,y) or (X,y,dir)
            if y.ndim == 4:
                y = y.squeeze(1)
            cond = X.to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=6)      # (B,H,W,C)
            y_onehot = y_onehot.permute(0,3,1,2).float() # (B,C,H,W)
            x = torch.randn_like(y_onehot)
            x = x.to(device)

            for t in reversed(range(0,T)):

              # sample z from a normal distribution if condition met
              if t > 0:
                  z = torch.randn_like(x)
              else:
                  z = 0

              # convert t to a tensor, send to the GPU, and expand its first dimension to be equal to batch size
              t = torch.tensor(t).to(device).expand(x.shape[0])

              # denoise x
              e = model(x, t, cond)

              x = reverse_diffusion(x, t, e, z, betas, alphas, alphas_bar)


            pred = torch.argmax(x, dim=1)
            #pred = torch.argmax(logits, dim=1)

            lt = y.detach().cpu().numpy().astype(np.int64)
            lp = pred.detach().cpu().numpy().astype(np.int64)

            for t, p in zip(lt, lp):
                hist += fast_hist(t.flatten(), p.flatten(), n_classes)

    # metrics (same as runningScore)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-12)
    mean_acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-12)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum() + 1e-12)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    scores = {
        "Pixel Acc: ": float(acc),
        "Class Accuracy: ": acc_cls,
        "Mean Class Acc: ": float(mean_acc_cls),
        "Freq Weighted IoU: ": float(fwavacc),
        "Mean IoU: ": float(mean_iu),
        "confusion_matrix": hist
    }
    cls_iou = dict(zip(range(n_classes), iu))
    return scores, cls_iou

@torch.no_grad()
def ddim_step(x, t, t_prev, eps_pred, alphas_bar, fix_batch):
    """
    Deterministic DDIM update (eta=0).
    x:        (B,C,H,W)
    t:        (B,) long
    t_prev:   (B,) long
    eps_pred: (B,C,H,W)
    """
    ab_t = fix_batch(alphas_bar, t, x.device)          # (B,1,1,1)
    ab_prev = fix_batch(alphas_bar, t_prev, x.device)  # (B,1,1,1)

    sqrt_ab_t = torch.sqrt(ab_t)
    sqrt_ab_prev = torch.sqrt(ab_prev)

    sqrt_1mab_t = torch.sqrt(1.0 - ab_t)
    sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev)

    # x0_hat from eps prediction (works for DDPM-trained eps models)
    x0_hat = (x - sqrt_1mab_t * eps_pred) / (sqrt_ab_t + 1e-8)

    # DDIM deterministic update
    x_prev = sqrt_ab_prev * x0_hat + sqrt_1mab_prev * eps_pred
    return x_prev


@torch.no_grad()
def sample_ddim(
    model,
    cond,                 # (B, cond_ch, H, W)
    steps=50,
    n_classes=6,
    T=1000,
    alphas_bar=None,
    fix_batch=None,
):
    """
    Fast DDIM sampling from noise -> segmentation channels.
    Returns:
      x: (B, C, H, W) final continuous tensor (logits-like)
      pred: (B, H, W) argmax segmentation
      conf: (B, H, W) max softmax prob
    """
    assert alphas_bar is not None, "alphas_bar is required"
    assert fix_batch is not None, "fix_batch is required"
    model.eval()

    device = cond.device
    B, _, H, W = cond.shape

    # Start from Gaussian noise in class-channel space
    x = torch.randn((B, n_classes, H, W), device=device)

    # Choose a reduced set of timesteps (linearly spaced)
    # Example: steps=50 -> 50 updates instead of 1000
    timesteps = torch.linspace(T - 1, 0, steps, device=device).long()

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i].item())
        t_prev_int = int(timesteps[i + 1].item())

        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        t_prev = torch.full((B,), t_prev_int, device=device, dtype=torch.long)

        eps_pred = model(x, t, cond)  # epsilon prediction
        x = ddim_step(x, t, t_prev, eps_pred, alphas_bar, fix_batch)

    # Final prediction + confidence
    probs = torch.softmax(x, dim=1)          # (B,C,H,W)
    conf = probs.max(dim=1).values           # (B,H,W)
    pred = probs.argmax(dim=1)               # (B,H,W)

    return x, pred, conf

def evaluate_metrics_diff_ddim(
    model,
    data_loader,
    device,
    alphas_bar,
    fix_batch,
    T=1000,
    n_classes=6,
    steps=50,          # DDIM steps: 50/100/200 are common
    max_batches=None,  # optional: set to 10 for quick debugging
):
    """
    Evaluates segmentation metrics using DDIM sampling.
    Returns:
      scores dict + per-class IoU dict
    """
    model.eval()
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)

    def fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        return np.bincount(
            n_class * label_true[mask] + label_pred[mask],
            minlength=n_class**2
        ).reshape(n_class, n_class)

    with torch.no_grad():
        for b_idx, batch in tqdm(enumerate(data_loader)):
            if (max_batches is not None) and (b_idx >= max_batches):
                break

            # batch can be (X,y) or (X,y,dir)
            X, y = batch[0], batch[1]
            if y.ndim == 4:
                y = y.squeeze(1)

            cond = X.to(device)
            y = y.to(device).long()  # (B,H,W)

            # DDIM sampling
            x_final, pred, conf = sample_ddim(
                model=model,
                cond=cond,
                steps=steps,
                n_classes=n_classes,
                T=T,
                alphas_bar=alphas_bar,
                fix_batch=fix_batch,
            )

            lt = y.detach().cpu().numpy().astype(np.int64)
            lp = pred.detach().cpu().numpy().astype(np.int64)

            for t_true, t_pred in zip(lt, lp):
                hist += fast_hist(t_true.flatten(), t_pred.flatten(), n_classes)

    # Compute metrics
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-12)
    mean_acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-12)
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / (hist.sum() + 1e-12)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    scores = {
        "Pixel Acc": float(acc),
        "Class Accuracy": acc_cls,
        "Mean Class Acc": float(mean_acc_cls),
        "Freq Weighted IoU": float(fwavacc),
        "Mean IoU": float(mean_iu),
        "confusion_matrix": hist,
        "ddim_steps": int(steps),
    }
    cls_iou = dict(zip(range(n_classes), iu))
    return scores, cls_iou