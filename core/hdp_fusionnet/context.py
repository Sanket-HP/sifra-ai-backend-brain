# core/hdp_fusionnet/context.py

import numpy as np

class ContextModule:
    """
    HDP-FusionNet Context Module.
    Determines context based on:
    - task type
    - dataset structure
    - data size
    - data variability
    """

    def __init__(self):
        print("[HDP-FUSIONNET] Context Module Loaded")

    def detect_context(self, goal, dataset):
        """
        Generates context vector:
        [task_type, rows, cols, variability_score]
        """

        ds = np.array(dataset)

        rows = ds.shape[0]
        cols = ds.shape[1] if len(ds.shape) > 1 else 1

        # variability = average std deviation
        if len(ds.shape) == 1:
            variability = float(np.std(ds))
        else:
            variability = float(np.mean(np.std(ds, axis=1)))

        # map goals to numeric context
        task_map = {
            "analyze": 1,
            "predict": 2,
            "forecast": 3,
            "anomaly": 4,
            "insights": 5
        }

        task_type = task_map.get(goal, 0)

        return [
            float(task_type),
            float(rows),
            float(cols),
            float(variability)
        ]
