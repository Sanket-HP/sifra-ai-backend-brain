# core/engine_router.py

from core.sifra_core import SifraCore

class EngineRouter:
    """
    Routes user goals to appropriate SIFRA AI engine functions.
    """

    def __init__(self):
        self.core = SifraCore()
        print("[ENGINE ROUTER] Ready.")

    def route(self, goal, dataset):
        """
        Automatically selects internal engine to execute.
        """
        print(f"[ROUTER] Received goal: {goal}")

        # Normalize goal
        goal = goal.lower().strip()

        # ---- Analysis ----
        if goal in ["analyze", "analysis", "auto_analyze"]:
            return self.core.run("analyze", dataset)

        # ---- Prediction ----
        if goal in ["predict", "prediction", "auto_predict"]:
            return self.core.run("predict", dataset)

        # ---- Forecast ----
        if goal in ["forecast", "future", "auto_forecast"]:
            return self.core.run("forecast", dataset)

        # ---- Anomaly Detection ----
        if goal in ["anomaly", "anomalies", "auto_anomaly"]:
            return self.core.run("anomaly", dataset)

        # ---- Insights ----
        if goal in ["insights", "auto_insights", "insight"]:
            return self.core.run("insights", dataset)

        # ---- Trend Extraction ----
        if goal in ["trend", "pattern", "statistics"]:
            return self.core.analyze_data(dataset)

        # ---- Default ----
        print("[ROUTER] Unknown goal. Returning error.")

        return {
            "error": "Unknown task",
            "goal": goal
        }
