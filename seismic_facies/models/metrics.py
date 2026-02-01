import numpy as np
import torch
import matplotlib.pyplot as plt

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
