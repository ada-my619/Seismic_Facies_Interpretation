import builtins
import torch
import matplotlib.pyplot as plt
from itertools import islice

def plot_segmentation_batch(
    model,
    data_loader,
    device,
    batch_idx=0,          
    n_classes=6,
    n_samples=4,
    cmap_seis="seismic",
    cmap_label="tab20",
    show_confidence=True,
    vmax_seis=None
):
    model.eval()

    # get the batch at batch_idx
    try:
        batch = next(islice(iter(data_loader), batch_idx, None))
    except StopIteration:
        raise ValueError(f"batch_idx={batch_idx} out of range for data_loader")

    if len(batch) == 2:
        X, y = batch
        direction = None
    else:
        X, y, direction = batch

    if y.ndim == 4:
        y = y.squeeze(1)

    with torch.no_grad():
        logits = model(X.to(device)).cpu()
        pred = torch.argmax(logits, dim=1)

        if show_confidence:
            probs = torch.softmax(logits, dim=1)
            conf = probs.max(dim=1).values

    B = X.shape[0]
    n = builtins.min(int(n_samples), int(B))
    ncols = 4 if show_confidence else 3

    fig, axes = plt.subplots(n, ncols, figsize=(4*ncols, 3*n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        x_i = X[i, 0]
        y_i = y[i]
        p_i = pred[i]

        dir_str = "" if direction is None else f"{direction[i]}"
        dir_str = " Inline" if dir_str == "i" else " Cross Line" if dir_str == "x" else dir_str

        ax = axes[i][0]
        im0 = ax.imshow(x_i.permute(1, 0), cmap=cmap_seis, aspect="auto", vmax=vmax_seis)
        ax.set_title(f"Seismic{dir_str}")
        ax.axis("off")
        plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[i][1]
        im1 = ax.imshow(y_i.permute(1, 0), cmap=cmap_label, aspect="auto",
                         vmin=0, vmax=n_classes-1)
        ax.set_title(f"Ground Truth{dir_str}")
        ax.axis("off")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[i][2]
        im2 = ax.imshow(p_i.permute(1, 0), cmap=cmap_label, aspect="auto",
                         vmin=0, vmax=n_classes-1)
        ax.set_title(f"Prediction{dir_str}")
        ax.axis("off")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        if show_confidence:
            ax = axes[i][3]
            im3 = ax.imshow(conf[i].permute(1, 0), aspect="auto")
            ax.set_title(f"Confidence{dir_str}")
            ax.axis("off")
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_sample_by_index(model, dataset, device, idx, n_classes=6, title=""):
    model.eval()

    item = dataset[idx]
    if len(item) == 2:
        x, y = item
        direction = None
    else:
        x, y, direction = item

    x = x.unsqueeze(0).to(device)
    if y.ndim == 3:
        y = y.squeeze(0)

    with torch.no_grad():
        logits = model(x).cpu()
        pred = torch.argmax(logits, dim=1)[0]

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(title)
    axs[0].imshow(x[0,0].cpu().permute(1,0), cmap="seismic", aspect="auto")
    axs[0].set_title("Seismic")
    axs[1].imshow(y.permute(1,0), cmap="tab20", vmin=0, vmax=n_classes-1)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred.permute(1,0), cmap="tab20", vmin=0, vmax=n_classes-1)
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.show()