# core/hds_unity/variation_channel.py

import numpy as np

class VariationChannel:
    """
    Variation Channel measures volatility or spread within each row.
    Higher variation = more unstable pattern.
    """

    def __init__(self):
        print("[HDS-UNITY] Variation Channel Loaded")

    def compute_variation(self, dataset):
        ds = np.array(dataset)

        if len(ds.shape) == 1:
            ds = ds.reshape(1, -1)

        # Standard deviation for each row → average
        variations = np.std(ds, axis=1)

        return float(np.mean(variations))
