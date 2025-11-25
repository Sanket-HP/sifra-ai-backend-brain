# core/hdp_fusionnet/emotion.py

import numpy as np

class EmotionModule:
    """
    Detects 'data emotion' — instability or volatility in dataset.
    This is not human emotion; it's pattern volatility.
    """

    def __init__(self):
        print("[HDP-FUSIONNET] Emotion Module Loaded")

    def detect_emotion(self, dataset):
        """
        Returns a single emotion score:
        - higher = more unstable/volatile
        - lower = smoother patterns
        """

        ds = np.array(dataset)

        if len(ds.shape) == 1:
            diffs = np.diff(ds)
        else:
            diffs = np.diff(ds, axis=1).flatten()

        if len(diffs) == 0:
            return 0.0

        volatility = float(np.std(diffs))

        # convert into normalized 0–1 scale
        score = min(1.0, volatility / 10)

        return score
