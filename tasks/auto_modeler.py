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
    Supports:
      - Regression
      - Classification
      - Clustering
    """

    def __init__(self):
        print("[TASK] Auto Modeler Engine Ready")

    # --------------------------------------------------------------
    # Detect problem type automatically
    # --------------------------------------------------------------
    def detect_task_type(self, y):
        y = np.array(y)

        # Categorical if few unique values & integers
        unique_vals = len(np.unique(y))

        if unique_vals <= 5:
            return "classification"

        # Regression if many numeric values
        if np.issubdtype(y.dtype, np.number):
            return "regression"

        return "clustering"

    # --------------------------------------------------------------
    # Handle both (X, y) OR dataset=[..., ..., ...]
    # --------------------------------------------------------------
    def parse_input(self, *args):
        """
        Supports two input formats:
        1️⃣ run(X, y)
        2️⃣ run(dataset) where last column = y
        """

        # Case 1: run(X, y)
        if len(args) == 2:
            X, y = args
            X = np.array(X)
            y = np.array(y)
            return X, y

        # Case 2: run(dataset)
        elif len(args) == 1:
            dataset = args[0]
            df = pd.DataFrame(dataset)

            if df.shape[1] < 2:
                raise ValueError("Dataset must have at least 2 columns (features + target).")

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y

        else:
            raise ValueError("Invalid input format for AutoModeler.run().")

    # --------------------------------------------------------------
    # MAIN TRAINING ENGINE
    # --------------------------------------------------------------
    def run(self, *args):
        try:
            X, y = self.parse_input(*args)
        except Exception as e:
            return {"error": str(e)}

        task = self.detect_task_type(y)

        # ---------------------------
        # 1️⃣ REGRESSION
        # ---------------------------
        if task == "regression":
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                models = {
                    "LinearRegression": LinearRegression(),
                    "RandomForestRegressor": RandomForestRegressor(n_estimators=120)
                }

                best_model = None
                best_score = -999
                results = {}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = r2_score(y_test, preds)
                    results[name] = float(score)

                    if score > best_score:
                        best_model = name
                        best_score = score

                return {
                    "status": "success",
                    "task_type": "regression",
                    "best_model": best_model,
                    "scores": results,
                    "r2_best": float(best_score)
                }

            except Exception as e:
                return {"error": f"Regression failed: {str(e)}"}

        # ---------------------------
        # 2️⃣ CLASSIFICATION
        # ---------------------------
        elif task == "classification":
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                models = {
                    "LogisticRegression": LogisticRegression(max_iter=300),
                    "RandomForestClassifier": RandomForestClassifier(n_estimators=120)
                }

                best_model = None
                best_score = -999
                results = {}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    results[name] = float(score)

                    if score > best_score:
                        best_model = name
                        best_score = score

                return {
                    "status": "success",
                    "task_type": "classification",
                    "best_model": best_model,
                    "scores": results,
                    "accuracy_best": float(best_score)
                }

            except Exception as e:
                return {"error": f"Classification failed: {str(e)}"}

        # ---------------------------
        # 3️⃣ CLUSTERING
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
# -----------------------------------------------------------
# END OF FILE   # -----------------------------------------------------------