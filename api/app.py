# api/app.py

from flask import Flask, request, jsonify

# Task Engines
from tasks.auto_analyze import AutoAnalyze
from tasks.auto_predict import AutoPredict
from tasks.auto_forecast import AutoForecast
from tasks.auto_anomaly import AutoAnomaly
from tasks.auto_insights import AutoInsights

# Utilities
from data.dataset_loader import DatasetLoader
from core.engine_router import EngineRouter

app = Flask(__name__)

# Loaders and Engines
loader = DatasetLoader()
router = EngineRouter()

analyzer = AutoAnalyze()
predictor = AutoPredict()
forecaster = AutoForecast()
anomaly_detector = AutoAnomaly()
insight_engine = AutoInsights()

# ----------------------- ROUTES -----------------------

@app.get("/")
def home():
    return {"status": "SIFRA AI API Running", "version": "1.0.0"}


# ------------- Common dataset extraction -----------

def extract_dataset(req):
    """ Helper to extract and validate dataset from JSON """
    if not req.json or "dataset" not in req.json:
        return None, {"error": "Missing 'dataset' in request"}, 400
    try:
        dataset = loader.load_raw(req.json["dataset"])
        return dataset, None, None
    except Exception as e:
        return None, {"error": f"Invalid dataset: {str(e)}"}, 400


# -------------------- ANALYZE -----------------------

@app.post("/analyze")
def analyze():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code
    result = analyzer.run(dataset)
    return jsonify(result)


# -------------------- PREDICT -----------------------

@app.post("/predict")
def predict():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code
    result = predictor.run(dataset)
    return jsonify(result)


# -------------------- FORECAST -----------------------

@app.post("/forecast")
def forecast():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code

    steps = request.json.get("steps", 5)
    try:
        steps = int(steps)
    except:
        steps = 5

    result = forecaster.run(dataset, steps)
    return jsonify(result)


# -------------------- ANOMALY -----------------------

@app.post("/anomaly")
def anomaly():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code
    result = anomaly_detector.run(dataset)
    return jsonify(result)


# -------------------- INSIGHTS -----------------------

@app.post("/insights")
def insights():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code
    result = insight_engine.run(dataset)
    return jsonify(result)


# -------------------- TREND -----------------------

@app.post("/trend")
def trend():
    dataset, err, code = extract_dataset(request)
    if err:
        return err, code
    result = router.route("trend", dataset)
    return jsonify({"trend_score": result})


# -------------------- SERVER START -----------------------

if __name__ == "__main__":
    print("[API] Starting SIFRA AI Server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
# --- IGNORE ---