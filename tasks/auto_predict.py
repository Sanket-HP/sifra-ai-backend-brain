# tasks/auto_predict.py

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

class AutoPredict:
    """
    Predicts the next value based on trend + variation + correlation.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Predict Module Ready")

    def run(self, dataset):
        print("\n[AUTO PREDICT] Running Prediction...")

        clean_data = self.preprocessor.clean(dataset)

        # Run full brain pipeline (intent = predict)
        result = self.core.run("predict", clean_data)

        # Prediction logic: last_value + trend
        last_val = clean_data.mean(axis=1).mean()
        trend = result["analysis_result"]["trend_score"]

        prediction = last_val + trend

        return {
            "task": "auto_predict",
            "intent": result["intent_vector"],
            "trend": trend,
            "prediction": prediction
        }
if __name__ == "__main__":
    auto_predict = AutoPredict()
    sample_data = {
        "feature1": [10, 12, 14, 16, 18],
        "feature2": [20, 22, 24, 26, 28]
    }
    prediction_result = auto_predict.run(sample_data)
    print("\nPrediction Result:", prediction_result)