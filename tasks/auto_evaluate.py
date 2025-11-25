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
    # Detect task type based on y_true
    # --------------------------------------------------------------
    def detect_type(self, y_true):
        y_true = np.array(y_true)
        unique_vals = len(np.unique(y_true))

        # Classification if few unique categories
        if unique_vals <= 5:
            return "classification"

        # Regression if numeric continuous
        if np.issubdtype(y_true.dtype, np.number):
            return "regression"

        # Fallback
        return "clustering"

    # --------------------------------------------------------------
    # Main evaluation
    # --------------------------------------------------------------
    def run(self, y_true, y_pred):
        """
        y_true: list / array of actual values
        y_pred: list / array of predicted values OR cluster labels
        """

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        task_type = self.detect_type(y_true)

        # -----------------------------
        # 1️⃣ REGRESSION EVALUATION
        # -----------------------------
        if task_type == "regression":
            try:
                return {
                    "status": "success",
                    "task_type": "regression",
                    "r2_score": float(r2_score(y_true, y_pred)),
                    "mse": float(mean_squared_error(y_true, y_pred)),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                }
            except Exception as e:
                return {"error": f"Regression evaluation failed: {str(e)}"}

        # -----------------------------
        # 2️⃣ CLASSIFICATION EVALUATION
        # -----------------------------
        elif task_type == "classification":
            try:
                # Prevent undefined precision/recall warnings
                precision = precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

                return {
                    "status": "success",
                    "task_type": "classification",
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                }
            except Exception as e:
                return {"error": f"Classification evaluation failed: {str(e)}"}

        # -----------------------------
        # 3️⃣ CLUSTERING EVALUATION
        # -----------------------------
        else:
            try:
                # Clustering silhouette requires X and labels
                # Here y_true acts as "features" only when clustering
                score = silhouette_score(y_true.reshape(-1, 1), y_pred)
                return {
                    "status": "success",
                    "task_type": "clustering",
                    "silhouette_score": float(score),
                }
            except Exception:
                return {
                    "status": "partial",
                    "task_type": "clustering",
                    "message": "Silhouette score not computable — returning labels.",
                    "labels": y_pred.tolist(),
                }
