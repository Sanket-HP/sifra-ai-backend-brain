# tasks/auto_insights.py

import numpy as np
from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

class AutoInsights:
    """
    Generates insights from dataset based on trends, variation, peaks, patterns.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Insights Module Ready")

    def run(self, dataset):
        print("\n[AUTO INSIGHTS] Extracting insights...")

        clean_data = self.preprocessor.clean(dataset)

        # Brain pipeline
        result = self.core.run("insights", clean_data)

        avg = float(clean_data.mean())
        trend = float(result["analysis_result"]["trend_score"])

        max_val = float(clean_data.max())
        min_val = float(clean_data.min())

        insights = [
            f"Overall average value is {avg:.2f}",
            f"General trend direction is {'upward' if trend > 0 else 'downward'}",
            f"Maximum observed value is {max_val}",
            f"Minimum observed value is {min_val}",
            f"Dataset volatility is {float(clean_data.std()):.2f}"
        ]

        return {
            "task": "auto_insights",
            "intent": result["intent_vector"],
            "trend": trend,
            "avg": avg,
            "max": max_val,
            "min": min_val,
            "insights": insights
        }
if __name__ == "__main__":      
    auto_insights = AutoInsights()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    insights_result = auto_insights.run(sample_data)
    print("\nInsights Result:", insights_result)