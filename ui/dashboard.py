# ui/dashboard.py

class Dashboard:
    """
    Terminal-based dashboard for SIFRA AI.
    """

    def __init__(self):
        print("[UI] SIFRA AI Dashboard Ready")

    def show_menu(self):
        print("\n========== SIFRA AI DASHBOARD ==========")
        print("1. Auto Analyze")
        print("2. Auto Predict")
        print("3. Auto Forecast")
        print("4. Auto Anomaly Detection")
        print("5. Auto Insights")
        print("6. Trend Extraction")
        print("7. Load Dataset")
        print("8. Exit")
        print("========================================")

    # --- ANALYSIS ---
    def show_analysis_result(self, result):
        print("\n========== ANALYSIS RESULT ==========")
        print("Task:", result.get("task"))
        print("Intent Vector:", result.get("intent_used"))
        print("Trend Score:", result["analysis_result"].get("trend_score"))
        print("======================================")

    # --- PREDICT ---
    def show_prediction_result(self, result):
        print("\n========== PREDICTION RESULT ==========")
        print("Predicted Value:", result.get("prediction"))
        print("Detected Trend:", result.get("trend"))
        print("========================================")

    # --- FORECAST ---
    def show_forecast_result(self, result):
        print("\n========== FORECAST RESULT ==========")
        print("Forecast Steps:", result.get("forecast_steps"))
        print("Forecasted Values:", result.get("forecast_values"))
        print("Trend:", result.get("trend"))
        print("=======================================")

    # --- ANOMALY ---
    def show_anomaly_result(self, result):
        print("\n========== ANOMALY DETECTION ==========")
        print("Mean:", result.get("mean"))
        print("Std Dev:", result.get("std"))
        print("Anomalies Found:")
        for a in result.get("anomalies_found", []):
            print(f" - Index {a['index']}: Value {a['value']}")
        print("========================================")

    # --- INSIGHTS ---
    def show_insights(self, result):
        print("\n========== INSIGHTS ==========")
        for insight in result.get("insights", []):
            print("•", insight)
        print("================================")

    # --- GENERIC MESSAGE ---
    def show_message(self, message):
        print("\n[UI MESSAGE]:", message)
        for insight in result["insights"]:
            print("•", insight)
        print("=========================")