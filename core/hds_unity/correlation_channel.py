# core/hds_unity/correlation_channel.py

import numpy as np

class CorrelationChannel:
    """
    Computes correlation strength inside each row.
    Converts a feature row into a correlation score.
    """

    def __init__(self):
        print("[HDS-UNITY] Correlation Channel Loaded")

    def compute_correlation(self, dataset):
        """
        For each row, compute correlation with a natural sequence [0,1,2,...].
        Returns average correlation score.
        """

        ds = np.array(dataset)

        # If 1D → reshape to 2D
        if len(ds.shape) == 1:
            ds = ds.reshape(1, -1)

        corr_scores = []

        for row in ds:
            seq = np.arange(len(row))

            # If row contains only 1 unique value, correlation is 0
            if np.std(row) == 0:
                corr_scores.append(0)
                continue

            corr = np.corrcoef(row, seq)[0, 1]
            corr_scores.append(corr)

        return float(np.mean(corr_scores))
