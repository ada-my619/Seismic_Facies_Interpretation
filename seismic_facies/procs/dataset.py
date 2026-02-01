import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from torch.utils.data import Subset


class SeismicSliceDataset(Dataset):
    def __init__(self, seismic_path, labels_path, axis=0, transform=None, mode="single", mmap_mode="r"):
        """
        mode:
          - "single": only inline/crosslinge slices along the specified axis
          - "both":  return inline and crossline slices (axis is ignored)
                    inline  => seis[i, :, :]  (X, T)
                    xline   => seis[:, x, :]  (I, T)
        """
        seismic_facies_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
        )

        data_dir = os.path.join(seismic_facies_root, "data/data")
        os.makedirs(data_dir, exist_ok=True)

        data_path = os.path.join(data_dir, seismic_path)
        label_path = os.path.join(data_dir, labels_path)
    
        self.seis = np.load(data_path, mmap_mode=mmap_mode)
        self.lab  = np.load(label_path,  mmap_mode=mmap_mode)
        assert self.seis.shape == self.lab.shape

        self.axis = axis
        self.transform = transform
        self.mode = mode

        I, X, T = self.seis.shape
        self.I, self.X, self.T = I, X, T

        if self.mode == "single":
            self.n = self.seis.shape[self.axis]
            self.index_map = None
        elif self.mode == "both":
            # build index list of (direction, index)
            # 'i' means inline index in [0..I-1]
            # 'x' means crossline index in [0..X-1]
            self.index_map = [("i", i) for i in range(I)] + [("x", x) for x in range(X)]
            self.n = len(self.index_map)
        else:
            raise ValueError("mode must be 'single' or 'both'")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.mode == "single":
            # inline or crossline slice along specified axis
            x = np.take(self.seis, idx, axis=self.axis)
            y = np.take(self.lab,  idx, axis=self.axis)

        else:
            # mode == "both"
            direction, k = self.index_map[idx]

            if direction == "i":        # inline slice
                x = self.seis[k, :, :]  # (X, T)
                y = self.lab[k, :, :]
            else:                       # crossline slice
                x = self.seis[:, k, :]  # (I, T)
                y = self.lab[:, k, :]

        # convert types
        x = torch.from_numpy(np.array(x)).float().unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(np.array(y)).long()                # (H,W)

        if self.transform:
            x, y = self.transform(x, y)

        # return direction for debugging/sampler
        if self.mode == "both":
            return x, y, direction

        return x, y

def split_indices_by_direction(base_dataset, val_frac=0.2, seed=42):
    assert hasattr(base_dataset, "index_map") and base_dataset.index_map is not None, \
        "base_dataset must be created with mode='both'"

    rng = random.Random(seed)

    inline_ids = [i for i, (d, _) in enumerate(base_dataset.index_map) if d == "i"]
    cross_ids  = [i for i, (d, _) in enumerate(base_dataset.index_map) if d == "x"]

    rng.shuffle(inline_ids)
    rng.shuffle(cross_ids)

    n_val_i = int(len(inline_ids) * val_frac)
    n_val_x = int(len(cross_ids)  * val_frac)

    val_indices   = inline_ids[:n_val_i] + cross_ids[:n_val_x]
    train_indices = inline_ids[n_val_i:] + cross_ids[n_val_x:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices

def create_train_val_datasets(base_dataset, val_frac=0.2, seed=42):
    train_indices, val_indices = split_indices_by_direction(base_dataset, val_frac=val_frac, seed=seed)

    train_ds = Subset(base_dataset, train_indices)
    val_ds   = Subset(base_dataset, val_indices)
    return train_ds, val_ds