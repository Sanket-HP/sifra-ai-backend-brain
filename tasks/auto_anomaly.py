# tasks/auto_anomaly.py

import numpy as np
from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

class AutoAnomaly:
    """
    Detects anomalies using variation & deviation logic.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Anomaly Detector Ready")

    def run(self, dataset):
        print("\n[AUTO ANOMALY] Detecting anomalies...")

        clean_data = self.preprocessor.clean(dataset)

        # Brain pipeline
        result = self.core.run("anomaly", clean_data)

        trend = float(result["analysis_result"]["trend_score"])
        avg = float(clean_data.mean())
        std = float(clean_data.std())

        anomalies = []

        for i, val in enumerate(clean_data.flatten()):
            if abs(val - avg) > 2 * std:
                anomalies.append({"index": int(i), "value": float(val)})

        return {
            "task": "auto_anomaly",
            "intent": result["intent_vector"],
            "trend": trend,
            "mean": avg,
            "std": std,
            "anomalies_found": anomalies
        }
if __name__ == "__main__":
    auto_anomaly = AutoAnomaly()
    sample_data = {
        "feature1": [10, 12, 14, 100, 18],
        "feature2": [20, 22, 24, 26, -50]
    }
    anomaly_result = auto_anomaly.run(sample_data)
    print("\nAnomaly Detection Result:", anomaly_result)