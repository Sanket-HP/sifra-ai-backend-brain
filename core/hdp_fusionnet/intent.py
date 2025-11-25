# core/hdp_fusionnet/intent.py

class IntentModule:
    """
    HDP-FusionNet intent engine.
    Converts user goals into numeric intent vectors understood by SIFRA Core.
    """

    def __init__(self):
        print("[HDP-FUSIONNET] Intent Module Loaded")

    # ---------------------------------------------------------
    # (NEW) Primary method required by SIFRA Core
    # ---------------------------------------------------------
    def detect_intent(self, goal):
        """
        Main intent decoder used by SIFRA Core.
        Maps a goal/task to a 5-dimensional intent vector.
        """

        goal = goal.lower().strip()

        mapping = {
            # ANALYSIS
            "analyze": [1, 0, 0, 0, 0],
            "analysis": [1, 0, 0, 0, 0],
            "auto_analyze": [1, 0, 0, 0, 0],

            # PREDICT
            "predict": [0, 1, 0, 0, 0],
            "prediction": [0, 1, 0, 0, 0],
            "auto_predict": [0, 1, 0, 0, 0],

            # FORECAST
            "forecast": [0, 0, 1, 0, 0],
            "future": [0, 0, 1, 0, 0],
            "auto_forecast": [0, 0, 1, 0, 0],

            # ANOMALY
            "anomaly": [0, 0, 0, 1, 0],
            "anomalies": [0, 0, 0, 1, 0],
            "auto_anomaly": [0, 0, 0, 1, 0],

            # INSIGHTS
            "insights": [0, 0, 0, 0, 1],
            "insight": [0, 0, 0, 0, 1],
            "auto_insights": [0, 0, 0, 0, 1],
        }

        return mapping.get(goal, [0, 0, 0, 0, 0])

    # ---------------------------------------------------------
    # (OLD) Legacy method (kept for compatibility)
    # ---------------------------------------------------------
    def extract_intent(self, goal):
        """
        Legacy method from earlier versions.
        Supports only analyze/predict/forecast.
        """

        goal = goal.lower().strip()

        if goal == "analyze":
            return [1, 0, 0]

        if goal == "predict":
            return [0, 1, 0]

        if goal == "forecast":
            return [0, 0, 1]

        return [0, 0, 0]
# Singleton instance
hdp_fusionnet_intent = IntentModule()