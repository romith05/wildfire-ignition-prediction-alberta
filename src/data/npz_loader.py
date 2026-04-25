import numpy as np


def load_npz_patch(npz_path: str, label_key: str = "class", channel_stats=None):
    arrs = np.load(npz_path)

    feature_keys = [k for k in arrs.files if k != label_key]
    feature_keys = sorted(feature_keys)

    X = np.stack([arrs[k] for k in feature_keys], axis=-1).astype("float32")

    if channel_stats is not None:
        mean = np.asarray(channel_stats["mean"])[: X.shape[-1]]
        std = np.asarray(channel_stats["std"])[: X.shape[-1]]
        X = (X - mean) / (std + 1e-6)

    y = arrs[label_key].astype("float32") if label_key in arrs.files else None

    return X, y, feature_keys
