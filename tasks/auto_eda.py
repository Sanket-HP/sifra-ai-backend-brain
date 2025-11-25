# tasks/auto_eda.py

import numpy as np
import pandas as pd

class AutoEDA:
    """
    Automated Exploratory Data Analysis Engine for SIFRA AI.
    Returns complete dataset insights as JSON.
    """

    def __init__(self):
        print("[TASK] Auto EDA Engine Ready")

    # -----------------------------------------
    # Detect outliers using IQR
    # -----------------------------------------
    def detect_outliers(self, data):
        Q1 = np.nanpercentile(data, 25)
        Q3 = np.nanpercentile(data, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = []
        for i, val in enumerate(data):
            if val < lower or val > upper:
                outliers.append({"index": i, "value": float(val)})
        
        return outliers

    # -----------------------------------------
    # Main EDA Function
    # -----------------------------------------
    def run(self, dataset):
        """
        Accepts Python list or NumPy array and performs full EDA.
        """

        df = pd.DataFrame(dataset)

        results = {}
        summary = {}

        # Convert all columns to numeric when possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Basic info
        summary["shape"] = df.shape
        summary["columns"] = df.columns.tolist()
        summary["missing_values"] = df.isna().sum().to_dict()
        summary["missing_ratio"] = (df.isna().mean()).round(4).to_dict()

        # Stats for each column
        numeric_stats = {}
        for col in df.columns:
            series = df[col].dropna()

            if series.empty:
                continue

            numeric_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "outliers": self.detect_outliers(series.values)
            }

        summary["column_statistics"] = numeric_stats

        # Correlation matrix (if >1 column)
        if df.shape[1] > 1:
            summary["correlation_matrix"] = df.corr().round(4).fillna(0).to_dict()
        else:
            summary["correlation_matrix"] = "Not enough columns for correlation"

        # Overall dataset summary
        results["summary"] = summary

        return {
            "status": "success",
            "eda_report": results
        }
