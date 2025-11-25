# tasks/auto_forecast.py

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

class AutoForecast:
    """
    Forecasts multiple future points using trend continuation.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Forecast Module Ready")

    def run(self, dataset, steps=5):
        print("\n[AUTO FORECAST] Running Forecast...")

        clean_data = self.preprocessor.clean(dataset)

        # Brain pipeline
        result = self.core.run("forecast", clean_data)

        trend = result["analysis_result"]["trend_score"]
        last_value = clean_data.mean(axis=1).mean()

        future = []
        curr = last_value

        for i in range(steps):
            curr += trend
            future.append(curr)

        return {
            "task": "auto_forecast",
            "intent": result["intent_vector"],
            "trend": trend,
            "forecast_steps": steps,
            "forecast_values": future
        }
if __name__ == "__main__":
    auto_forecast = AutoForecast()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    forecast_result = auto_forecast.run(sample_data, steps=5)
    print("\nForecast Result:", forecast_result)