# core/hds_unity/memory_signature.py

import numpy as np

class MemorySignature:
    """
    Memory Signature compresses Fusion Matrix into a stable pattern signature.
    This allows SIFRA AI to compare datasets intelligently.
    """

    def __init__(self):
        print("[HDS-UNITY] Memory Signature Module Loaded")

    def generate_signature(self, fusion_vector):
        """
        Creates a single signature number:
        mean + variance combination
        """

        mean_val = fusion_vector.mean()
        var_val  = np.var(fusion_vector)

        # Signature = weighted combination
        return float((mean_val * 0.7) + (var_val * 0.3))
