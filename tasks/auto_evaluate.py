# tasks/auto_evaluate.py

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score
)

class AutoEvaluate:
    """
    Autonomous model evaluation engine for SIFRA AI.
    Supports regression, classification, clustering.
    """

    def __init__(self):
        print("[TASK] Auto Evaluation Engine Ready")

    # --------------------------------------------------------------
    # Detect task type based on y / labels
    # --------------------------------------------------------------
    def detect_type(self, y_true):
        unique_vals = len(np.unique(y_true))

        if unique_vals <= 5:
            return "classification"

        if np.issubdtype(np.array(y_true).dtype, np.number):
            return "regression"

        return "clustering"

    # --------------------------------------------------------------
    # Main evaluation entry
    # --------------------------------------------------------------
    def run(self, y_true, y_pred):
        """
        y_true: list/array of actual values
        y_pred: list/array of predicted values or cluster IDs
        """

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        task_type = self.detect_type(y_true)

        # ---------------------------
        # 1️⃣ REGRESSION METRICS
        # ---------------------------
        if task_type == "regression":
            try:
                return {
                    "status": "success",
                    "task_type": "regression",
                    "r2_score": float(r2_score(y_true, y_pred)),
                    "mse": float(mean_squared_error(y_true, y_pred)),
                    "mae": float(mean_absolute_error(y_true, y_pred))
                }
            except Exception as e:
                return {"error": f"Regression evaluation failed: {str(e)}"}

        # ---------------------------
        # 2️⃣ CLASSIFICATION METRICS
        # ---------------------------
        elif task_type == "classification":
            try:
                return {
                    "status": "success",
                    "task_type": "classification",
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, average='weighted')),
                    "recall": float(recall_score(y_true, y_pred, average='weighted')),
                    "f1_score": float(f1_score(y_true, y_pred, average='weighted'))
                }
            except Exception as e:
                return {"error": f"Classification evaluation failed: {str(e)}"}

        # ---------------------------
        # 3️⃣ CLUSTERING METRICS
        # ---------------------------
        else:
            try:
                score = silhouette_score(y_true.reshape(-1, 1), y_pred)
                return {
                    "status": "success",
                    "task_type": "clustering",
                    "silhouette_score": float(score)
                }
            except:
                return {
                    "status": "partial",
                    "message": "Silhouette score not computable — returning labels only.",
                    "labels": y_pred.tolist()
                }
