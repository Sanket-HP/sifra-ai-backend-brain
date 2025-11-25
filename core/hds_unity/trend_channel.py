# core/hds_unity/trend_channel.py

import numpy as np

class TrendChannel:
    """
    Computes trend using a simple slope formula.
    Works even for single-column datasets.
    """

    def __init__(self):
        print("[HDS-UNITY] Trend Channel Module Loaded")

    def compute_trend(self, data):
        """
        Computes basic upward/downward trend using linear regression.
        Prevents NaN issues for single-column data.
        """

        data = np.array(data, dtype=float).flatten()

        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]  # slope = trend

        return float(slope)
