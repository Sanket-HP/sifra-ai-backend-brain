from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# -----------------------------------------------------------
# LAZY LOADING (Fixes Vercel cold-start crash issues)
# -----------------------------------------------------------
def load_engines():
    # Core & preprocessing
    from data.dataset_loader import DatasetLoader
    from core.engine_router import EngineRouter

    # Old Engines
    from tasks.auto_analyze import AutoAnalyze
    from tasks.auto_predict import AutoPredict
    from tasks.auto_forecast import AutoForecast
    from tasks.auto_anomaly import AutoAnomaly
    from tasks.auto_insights import AutoInsights

    # New Engines
    from tasks.auto_visualize import AutoVisualize
    from tasks.auto_eda import AutoEDA
    from tasks.auto_feature_engineering import AutoFeatureEngineering
    from tasks.auto_modeler import AutoModeler
    from tasks.auto_evaluate import AutoEvaluate
    from tasks.auto_bigdata import AutoBigData

    engines = {
        # Loader & Router
        "loader": DatasetLoader(),
        "router": EngineRouter(),

        # Old Modules
        "analyzer": AutoAnalyze(),
        "predictor": AutoPredict(),
        "forecaster": AutoForecast(),
        "anomaly": AutoAnomaly(),
        "insight": AutoInsights(),

        # New Modules
        "visualize": AutoVisualize(),
        "eda": AutoEDA(),
        "feature_eng": AutoFeatureEngineering(),
        "modeler": AutoModeler(),
        "evaluate": AutoEvaluate(),
        "bigdata": AutoBigData(),
    }

    return engines


# -----------------------------------------------------------
# HOME ROUTE
# -----------------------------------------------------------
@app.get("/")
def home():
    return {"status": "SIFRA AI API Running", "version": "2.0.0"}


# -----------------------------------------------------------
# EXTRACT DATASET
# -----------------------------------------------------------
def extract_dataset(req, loader):
    if not req.json or "dataset" not in req.json:
        return None, {"error": "Missing 'dataset' in request"}, 400

    try:
        dataset = loader.load_raw(req.json["dataset"])
        return dataset, None, None
    except Exception as e:
        return None, {"error": f"Invalid dataset: {str(e)}"}, 400


# -----------------------------------------------------------
# OLD ENGINE ROUTES
# -----------------------------------------------------------

@app.post("/analyze")
def analyze():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["analyzer"].run(dataset))


@app.post("/predict")
def predict():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["predictor"].run(dataset))


@app.post("/forecast")
def forecast():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    steps = request.json.get("steps", 5)
    try:
        steps = int(steps)
    except:
        steps = 5

    return jsonify(engines["forecaster"].run(dataset, steps))


@app.post("/anomaly")
def anomaly():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["anomaly"].run(dataset))


@app.post("/insights")
def insights():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code
    return jsonify(engines["insight"].run(dataset))


@app.post("/trend")
def trend():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    score = engines["router"].route("trend", dataset)
    return jsonify({"trend_score": score})


# -----------------------------------------------------------
# NEW ENGINE ROUTES (Visualization, EDA, FE, Modeler, Eval, BigData)
# -----------------------------------------------------------

@app.post("/visualize")
def visualize():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    result = engines["visualize"].run(dataset)
    return jsonify(result)


@app.post("/eda")
def eda():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    result = engines["eda"].run(dataset)
    return jsonify(result)


@app.post("/feature_engineering")
def feature_engineering():
    engines = load_engines()
    dataset, err, code = extract_dataset(request, engines["loader"])
    if err:
        return err, code

    result = engines["feature_eng"].run(dataset)
    return jsonify(result)


@app.post("/modeler")
def modeler():
    engines = load_engines()
    data = request.json

    if "X" not in data or "y" not in data:
        return {"error": "Provide 'X' and 'y' for model training"}, 400

    result = engines["modeler"].run(data["X"], data["y"])
    return jsonify(result)


@app.post("/evaluate")
def evaluate():
    engines = load_engines()
    data = request.json

    if "y_true" not in data or "y_pred" not in data:
        return {"error": "Provide 'y_true' and 'y_pred' for evaluation"}, 400

    result = engines["evaluate"].run(data["y_true"], data["y_pred"])
    return jsonify(result)


@app.post("/bigdata")
def bigdata():
    engines = load_engines()

    if "file_path" not in request.json:
        return {"error": "Missing 'file_path'"}, 400

    file_path = request.json["file_path"]
    result = engines["bigdata"].run(file_path)
    return jsonify(result)


# ❗ DO NOT ADD app.run() — Vercel handles routing.

