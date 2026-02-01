import random
import numpy as np
import torch

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True

def select_device():
    """
    Select device for torch
    """
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = 'cuda'
    else:
        print("No GPU available!")

    return device

import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import builtins


def compute_class_weights(
    dataset,
    num_classes=None,
    ignore_index=-1,
    max_samples=None,
    verbose=True,
):
    """
    Compute class weights from a SeismicSliceDataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Your SeismicSliceDataset (single or both mode)
    num_classes : int, optional
        Total number of classes. If None, inferred from data.
    ignore_index : int
        Label value to ignore (e.g. -1 for no-data)
    max_samples : int, optional
        If set, only iterate over first N samples (for speed)
    verbose : bool

    Returns
    -------
    weights : torch.FloatTensor
        Shape (num_classes,)
    """

    counter = Counter()
    n = len(dataset) if max_samples is None else builtins.min(len(dataset), max_samples)

    for i in tqdm(range(n), disable=not verbose):
        item = dataset[i]

        # handle (x, y) or (x, y, direction)
        y = item[1]

        y_np = y.numpy().ravel()
        y_np = y_np[y_np != ignore_index]

        counter.update(y_np.tolist())

    if len(counter) == 0:
        raise RuntimeError("No valid labels found to compute class weights.")

    if num_classes is None:
        num_classes = builtins.max(counter.keys()) + 1

    counts = np.zeros(num_classes, dtype=np.float64)
    for cls, cnt in counter.items():
        if cls < num_classes:
            counts[cls] = cnt

    # avoid division by zero
    counts[counts == 0] = 1.0

    # inverse-frequency weighting
    weights = 1.0 / counts

    # normalize (optional but recommended)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)
