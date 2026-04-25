import numpy as np
from tensorflow.keras.utils import Sequence


class KerasNPZGenerator(Sequence):
    """
    Keras-compatible generator wrapping NPZPatchDataset.

    Returns:
    X: (batch, 64, 64, C)
    y: (batch, 64, 64, 1)
    """

    def __init__(self, dataset, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))

        batch_indices = self.indices[start:end]

        X_batch = []
        y_batch = []

        for idx in batch_indices:
            X, y = self.dataset[idx]
            X_batch.append(X)
            y_batch.append(y)

        return np.stack(X_batch).astype("float32"), np.stack(y_batch).astype("float32")

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
