import numpy as np


def load_npz_patch(npz_path: str, label_key: str = "class", channel_stats=None):
    """Load one NPZ patch with stable feature ordering and optional label.

    Returns:
        X: feature tensor shaped (H, W, C)
        y: label tensor shaped (H, W, 1), or None if missing
        feature_keys: sorted feature key list

    The label shape intentionally matches NPZPatchDataset so evaluation and
    inference utilities do not accidentally broadcast (H, W) labels against
    (H, W, 1) predictions.
    """
    arrs = np.load(npz_path)

    feature_keys = [k for k in arrs.files if k != label_key]
    feature_keys = sorted(feature_keys)

    X = np.stack([arrs[k] for k in feature_keys], axis=-1).astype("float32")

    if channel_stats is not None:
        mean = np.asarray(channel_stats["mean"])[: X.shape[-1]]
        std = np.asarray(channel_stats["std"])[: X.shape[-1]]
        X = (X - mean) / (std + 1e-6)

    if label_key in arrs.files:
        y = arrs[label_key].astype("float32")
        if y.ndim == 2:
            y = y[..., np.newaxis]
        elif y.ndim == 3 and y.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Expected label shape (H, W) or (H, W, 1), got {y.shape}")
    else:
        y = None

    return X, y, feature_keys
