import os
import glob
import numpy as np
from torch.utils.data import Dataset


class NPZPatchDataset(Dataset):
    """
    PyTorch-style dataset for wildfire NPZ patches.

    Expected NPZ structure:
    - Multiple feature arrays, each shaped (64, 64)
    - Label key: "class"
    - Output X shape: (64, 64, C)
    - Output y shape: (64, 64, 1)
    """

    def __init__(self, folder, label_key="class", transforms=None, channel_stats=None):
        self.folder = folder
        self.label_key = label_key
        self.transforms = transforms
        self.channel_stats = channel_stats

        self.files = sorted(glob.glob(os.path.join(folder, "patch_*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No patch files found in: {folder}")

        self.valid_files = []
        self.skipped_files = []

        for file_path in self.files:
            try:
                arrs = np.load(file_path)
                if self.label_key in arrs.files:
                    self.valid_files.append(file_path)
                else:
                    self.skipped_files.append(os.path.basename(file_path))
            except Exception:
                self.skipped_files.append(os.path.basename(file_path))

        if len(self.valid_files) == 0:
            raise ValueError(f"No valid NPZ files found in: {folder}")

        first = np.load(self.valid_files[0])
        self.feature_keys = [k for k in first.files if k != self.label_key]
        self.C = len(self.feature_keys)

        print(f"Loaded dataset: {folder}")
        print(f"Valid patches: {len(self.valid_files)}")
        print(f"Skipped patches: {len(self.skipped_files)}")
        print(f"Channels: {self.C}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        arrs = np.load(file_path)

        feature_keys = [k for k in arrs.files if k != self.label_key]

        X = np.stack([arrs[k] for k in feature_keys], axis=-1).astype(np.float32)
        y = arrs[self.label_key].astype(np.float32)

        if self.channel_stats is not None:
            mean = np.asarray(self.channel_stats["mean"])[: X.shape[-1]]
            std = np.asarray(self.channel_stats["std"])[: X.shape[-1]]
            X = (X - mean) / (std + 1e-6)

        if self.transforms is not None:
            transformed = self.transforms(image=X, mask=y)
            X = transformed["image"]
            y = transformed["mask"]

        y = y[..., np.newaxis]

        return X, y
