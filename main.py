# main.py

import json
from data.dataset_loader import DatasetLoader

# Old DS task engines
from tasks.auto_analyze import AutoAnalyze
from tasks.auto_predict import AutoPredict
from tasks.auto_forecast import AutoForecast
from tasks.auto_anomaly import AutoAnomaly
from tasks.auto_insights import AutoInsights

# New extended DS engines
from tasks.auto_visualize import AutoVisualize
from tasks.auto_eda import AutoEDA
from tasks.auto_feature_engineering import AutoFeatureEngineering
from tasks.auto_modeler import AutoModeler
from tasks.auto_evaluate import AutoEvaluate
from tasks.auto_bigdata import AutoBigData

from ui.dashboard import Dashboard
from core.engine_router import EngineRouter


def safe_eval(expr):
    """Safely evaluate user input to avoid crashes."""
    try:
        return eval(expr)
    except Exception:
        raise ValueError("Invalid input format. Use Python list or dict format.")


def main():

    print("\n===============================")
    print("     SIFRA AI - Autonomous")
    print("    Data Scientist Engine")
    print("===============================\n")

    dashboard = Dashboard()
    loader = DatasetLoader()

    # OLD engines
    analyzer = AutoAnalyze()
    predictor = AutoPredict()
    forecaster = AutoForecast()
    anomaly_detector = AutoAnomaly()
    insight_engine = AutoInsights()

    # NEW engines
    visualizer = AutoVisualize()
    eda_engine = AutoEDA()
    feature_engineer = AutoFeatureEngineering()
    model_engine = AutoModeler()
    evaluator = AutoEvaluate()
    bigdata_engine = AutoBigData()

    router = EngineRouter()

    while True:

        print("\n========== SIFRA AI DASHBOARD ==========")
        print(" 1. Auto Analyze")
        print(" 2. Auto Predict")
        print(" 3. Auto Forecast")
        print(" 4. Auto Anomaly Detection")
        print(" 5. Auto Insights")
        print(" 6. Trend Extraction")

        print(" 7. Auto Visualization")
        print(" 8. Auto EDA")
        print(" 9. Auto Feature Engineering")
        print("10. Auto Model Builder")
        print("11. Auto Evaluation")
        print("12. Auto Big Data Processing")

        print("13. Load Dataset File")
        print("14. Exit")
        print("========================================")

        choice = input("\nEnter choice: ")

        # 1️⃣ Auto Analyze
        if choice == "1":
            print("\n[INPUT] Enter dataset (Python list):")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = analyzer.run(dataset)
                dashboard.show_analysis_result(result)
            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # 2️⃣ Auto Predict
        elif choice == "2":
            print("\n[INPUT] Dataset for prediction")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = predictor.run(dataset)
                print("\n===== PREDICTION RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # 3️⃣ Auto Forecast
        elif choice == "3":
            print("\n[INPUT] Dataset for forecasting")
            data = input("Dataset: ")
            steps = input("Steps (default=5): ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                steps = int(steps) if steps else 5
                result = forecaster.run(dataset, steps)
                print("\n===== FORECAST RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Forecast error:", e)

        # 4️⃣ Auto Anomaly Detection
        elif choice == "4":
            print("\n[INPUT] Dataset for anomaly detection")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = anomaly_detector.run(dataset)
                print("\n===== ANOMALY REPORT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # 5️⃣ Auto Insights
        elif choice == "5":
            print("\n[INPUT] Dataset for insights")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = insight_engine.run(dataset)
                print("\n===== INSIGHTS =====")
                for line in result["insights"]:
                    print("-", line)
            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # 6️⃣ Trend Extraction
        elif choice == "6":
            print("\n[INPUT] Dataset for trend extraction")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                score = router.route("trend", dataset)
                print("\nTrend Score:", score)
            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # 7️⃣ Auto Visualization
        elif choice == "7":
            print("\n[INPUT] Dataset for visualization")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = visualizer.run(dataset)
                print("\n===== VISUALIZATION SPECS =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Visualization error:", e)

        # 8️⃣ Auto EDA
        elif choice == "8":
            print("\n[INPUT] Dataset for EDA")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = eda_engine.run(dataset)
                print("\n===== EDA REPORT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] EDA error:", e)

        # 9️⃣ Auto Feature Engineering
        elif choice == "9":
            print("\n[INPUT] Dataset for Feature Engineering")
            data = input("Dataset: ")
            try:
                dataset = loader.load_raw(safe_eval(data))
                result = feature_engineer.run(dataset)
                print("\n===== FEATURE ENGINEERING RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Feature engineering error:", e)

        # 🔟 Auto Model Builder
        elif choice == "10":
            print("\nEnter X (features): ")
            X = safe_eval(input("X: "))
            print("\nEnter y (labels): ")
            y = safe_eval(input("y: "))
            try:
                result = model_engine.run(X, y)
                print("\n===== MODEL RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Model building error:", e)

        # 1️⃣1️⃣ Auto Evaluate
        elif choice == "11":
            print("\nEnter true labels:")
            y_true = safe_eval(input("y_true: "))
            print("\nEnter predicted labels:")
            y_pred = safe_eval(input("y_pred: "))

            try:
                result = evaluator.run(y_true, y_pred)
                print("\n===== EVALUATION RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] Evaluation error:", e)

        # 1️⃣2️⃣ Auto Big Data Processing
        elif choice == "12":
            print("\nEnter big CSV file path:")
            path = input("File path: ").strip()

            try:
                result = bigdata_engine.run(path)
                print("\n===== BIGDATA RESULT =====")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("[ERROR] BigData processing error:", e)

        # 1️⃣3️⃣ Load dataset file
        elif choice == "13":
            print("\nChoose file type:")
            print("1. CSV")
            print("2. Excel")
            print("3. JSON")
            ft = input("File type: ")
            path = input("File path: ")

            try:
                if ft == "1":
                    data = loader.load_csv(path)
                elif ft == "2":
                    data = loader.load_excel(path)
                elif ft == "3":
                    data = loader.load_json(path)
                else:
                    print("[ERROR] Invalid format.")
                    continue

                print("\n[RESULT] Loaded dataset:")
                print(data)

            except Exception as e:
                print("[ERROR] Could not load file:", e)

        # 1️⃣4️⃣ Exit
        elif choice == "14":
            print("\n[EXIT] Shutting down SIFRA AI. Goodbye!")
            break

        else:
            print("\n[ERROR] Invalid choice, try again.")


if __name__ == "__main__":
    main()
