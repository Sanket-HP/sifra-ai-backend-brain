# tasks/auto_modeler.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    silhouette_score
)

class AutoModeler:
    """
    Autonomous ML Model Training Engine for SIFRA AI.
    Automatically detects task type:
      - Regression
      - Classification
      - Clustering
    And trains the best model.
    """

    def __init__(self):
        print("[TASK] Auto Modeler Engine Ready")

    # --------------------------------------------------------------
    # Detect problem type automatically
    # --------------------------------------------------------------
    def detect_task_type(self, y):
        unique_vals = len(np.unique(y))

        # Classification if target has few categories
        if unique_vals <= 5:
            return "classification"

        # Regression if target is numeric with many values
        if unique_vals > 5 and np.issubdtype(y.dtype, np.number):
            return "regression"

        # Otherwise clustering
        return "clustering"

    # --------------------------------------------------------------
    # Main training engine
    # --------------------------------------------------------------
    def run(self, dataset):
        """
        dataset must be 2D: [[x1, x2, ..., y], ...]
        Last column is treated as target (except clustering).
        """

        df = pd.DataFrame(dataset)

        if df.shape[1] < 2:
            return {"error": "Dataset requires at least 2 columns: features + target"}

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        task = self.detect_task_type(y)

        # ---------------------------
        # 1️⃣ REGRESSION
        # ---------------------------
        if task == "regression":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=150)
            }

            best_model = None
            best_score = -float("inf")
            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = r2_score(y_test, preds)
                results[name] = float(score)

                if score > best_score:
                    best_score = score
                    best_model = name

            return {
                "status": "success",
                "task_type": "regression",
                "best_model": best_model,
                "scores": results,
                "r2_best": float(best_score)
            }

        # ---------------------------
        # 2️⃣ CLASSIFICATION
        # ---------------------------
        elif task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "LogisticRegression": LogisticRegression(max_iter=300),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=150)
            }

            best_model = None
            best_score = -float("inf")
            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = accuracy_score(y_test, preds)
                results[name] = float(score)

                if score > best_score:
                    best_score = score
                    best_model = name

            return {
                "status": "success",
                "task_type": "classification",
                "best_model": best_model,
                "scores": results,
                "accuracy_best": float(best_score)
            }

        # ---------------------------
        # 3️⃣ CLUSTERING (unsupervised)
        # ---------------------------
        else:
            try:
                k = 3
                model = KMeans(n_clusters=k, random_state=42)
                model.fit(X)

                labels = model.labels_
                score = silhouette_score(X, labels)

                return {
                    "status": "success",
                    "task_type": "clustering",
                    "clusters": int(k),
                    "silhouette": float(score),
                    "labels": labels.tolist()
                }

            except Exception as e:
                return {"error": f"Clustering failed: {str(e)}"}
