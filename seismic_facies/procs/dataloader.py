import torch
from torch.utils.data import Sampler, Subset, DataLoader
import random

class DirectionBatchSampler(Sampler):
    """
    Works with:
      - a base dataset (must have .index_map)
      - a torch.utils.data.Subset of that dataset

    Guarantees each batch is 'i' only or 'x' only.
    """

    def __init__(self, ds_or_subset, batch_size, shuffle=True, seed=42):
        self.ds = ds_or_subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # Detect Subset vs base dataset
        if isinstance(ds_or_subset, Subset):
            self.is_subset = True
            base = ds_or_subset.dataset
            base_indices = list(ds_or_subset.indices)  # indices into base dataset
        else:
            self.is_subset = False
            base = ds_or_subset
            base_indices = list(range(len(ds_or_subset)))  # indices into base dataset (itself)

        if not hasattr(base, "index_map") or base.index_map is None:
            raise ValueError("Base dataset must have index_map (create dataset with mode='both').")

        # Group positions by direction.
        # IMPORTANT:
        # - if Subset: yield positions in subset (0..len(subset)-1)
        # - else: yield base indices directly
        self.inline_ids = []
        self.cross_ids = []

        for pos, base_idx in enumerate(base_indices):
            d, _ = base.index_map[base_idx]
            yield_id = pos if self.is_subset else base_idx

            if d == "i":
                self.inline_ids.append(yield_id)
            else:
                self.cross_ids.append(yield_id)

    def __iter__(self):
        inline = self.inline_ids.copy()
        cross  = self.cross_ids.copy()

        if self.shuffle:
            self.rng.shuffle(inline)
            self.rng.shuffle(cross)

        inline_batches = [inline[i:i+self.batch_size] for i in range(0, len(inline), self.batch_size)]
        cross_batches  = [cross[i:i+self.batch_size]  for i in range(0, len(cross),  self.batch_size)]

        all_batches = inline_batches + cross_batches
        if self.shuffle:
            self.rng.shuffle(all_batches)

        for b in all_batches:
            yield b

    def __len__(self):
        return (len(self.inline_ids) + len(self.cross_ids)) // self.batch_size


def create_dataloader_with_direction_batches(dataset, batch_size):
    sampler = DirectionBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    loader   = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=4)
    return loader