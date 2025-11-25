# core/hds_unity/fusion_matrix.py

import numpy as np

class FusionMatrix:
    """
    Fusion Matrix combines all pattern channels into a single representation.
    This acts as the 'reasoning surface' of HDS-Unity Engine.
    """

    def __init__(self):
        print("[HDS-UNITY] Fusion Matrix Module Loaded")

    def fuse(self, trend_score, corr_score, var_score):
        """
        Creates a fusion vector:
        [trend, correlation, variation]
        """

        return np.array([trend_score, corr_score, var_score], dtype=float)
