# main.py

from data.dataset_loader import DatasetLoader
from tasks.auto_analyze import AutoAnalyze
from tasks.auto_predict import AutoPredict
from tasks.auto_forecast import AutoForecast
from tasks.auto_anomaly import AutoAnomaly
from tasks.auto_insights import AutoInsights

from ui.dashboard import Dashboard
from core.engine_router import EngineRouter


def main():

    print("\n===============================")
    print("     SIFRA AI - Autonomous")
    print("    Data Scientist Engine")
    print("===============================")

    dashboard = Dashboard()
    loader = DatasetLoader()

    analyzer = AutoAnalyze()
    predictor = AutoPredict()
    forecaster = AutoForecast()
    anomaly_detector = AutoAnomaly()
    insight_engine = AutoInsights()

    router = EngineRouter()

    while True:
        dashboard.show_menu()
        choice = input("\nEnter choice: ")

        # Auto Analyze
        if choice == "1":
            print("\n[INPUT] Enter dataset as Python list (example: [[1,2,3],[4,5,6]])")
            data_input = input("Dataset: ")

            try:
                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = analyzer.run(dataset)
                dashboard.show_analysis_result(result)

            except Exception as e:
                print("[ERROR] Invalid dataset format:", e)

        # Auto Predict
        elif choice == "2":
            print("\n[INPUT] Enter dataset for prediction")
            data_input = input("Dataset: ")

            try:
                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = predictor.run(dataset)

                print("\n===== PREDICTION RESULT =====")
                print("Prediction:", result["prediction"])
                print("Trend Score:", result["trend"])
                print("=============================")

            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # Auto Forecast
        elif choice == "3":
            print("\n[INPUT] Enter dataset for forecasting")
            data_input = input("Dataset: ")

            try:
                steps = input("How many steps to forecast (default=5): ")
                steps = int(steps) if steps.strip() != "" else 5

                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = forecaster.run(dataset, steps)

                print("\n===== FORECAST RESULT =====")
                print("Forecasted Values:", result["forecast_values"])
                print("Trend Score:", result["trend"])
                print("============================")

            except Exception as e:
                print("[ERROR] Forecast error:", e)

        # Auto Anomaly
        elif choice == "4":
            print("\n[INPUT] Enter dataset for anomaly detection")
            data_input = input("Dataset: ")

            try:
                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = anomaly_detector.run(dataset)

                print("\n===== ANOMALY REPORT =====")
                print("Mean:", result["mean"])
                print("Std:", result["std"])
                print("Anomalies:", result["anomalies_found"])
                print("===========================")

            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # Auto Insights
        elif choice == "5":
            print("\n[INPUT] Enter dataset for insights")
            data_input = input("Dataset: ")

            try:
                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = insight_engine.run(dataset)

                print("\n===== INSIGHTS =====")
                for i in result["insights"]:
                    print("-", i)
                print("====================")

            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # Trend Extraction Direct
        elif choice == "6":
            print("\n[INPUT] Enter dataset for trend extraction")
            data_input = input("Dataset: ")

            try:
                dataset = eval(data_input)
                dataset = loader.load_raw(dataset)
                result = router.route("trend", dataset)
                print("\nTrend Score:", result)

            except Exception as e:
                print("[ERROR] Invalid dataset:", e)

        # Load Dataset File
        elif choice == "7":
            print("\n[DATA] Choose file type:")
            print("1. CSV")
            print("2. Excel")
            print("3. JSON")

            ft = input("File type: ")
            path = input("Enter file path: ")

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

        elif choice == "8":
            print("\n[EXIT] Shutting down Sifra AI. Goodbye!")
            break

        else:
            print("\n[ERROR] Invalid choice, try again.")


if __name__ == "__main__":
    main()
