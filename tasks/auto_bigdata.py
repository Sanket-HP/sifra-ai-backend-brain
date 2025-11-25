# tasks/auto_bigdata.py

import numpy as np
import pandas as pd
import os

class AutoBigData:
    """
    Lightweight Big Data Engine for SIFRA AI.
    Handles massive datasets using chunk streaming (memory-safe).
    """

    def __init__(self):
        print("[TASK] Auto BigData Engine Ready")

    # ------------------------------------------------------------
    # 0️⃣ Clean file path (Fix for Windows quotes "D:\file.csv")
    # ------------------------------------------------------------
    def clean_path(self, file_path):
        if isinstance(file_path, str):
            return file_path.strip().replace('"', '').replace("'", "")
        return file_path

    # ------------------------------------------------------------
    # 1️⃣ Stream a large CSV file safely (chunk by chunk)
    # ------------------------------------------------------------
    def stream_csv(self, file_path, chunk_size=50000):
        """
        Reads large CSV files in chunks to avoid memory overflow.
        """
        file_path = self.clean_path(file_path)

        if not os.path.exists(file_path):
            print(f"[BIGDATA ERROR] File not found: {file_path}")
            return

        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                yield chunk
        except Exception as e:
            print(f"[BIGDATA ERROR] {str(e)}")
            return

    # ------------------------------------------------------------
    # 2️⃣ Incremental statistics for massive files
    # ------------------------------------------------------------
    def incremental_stats(self, file_path, chunk_size=50000):
        """
        Calculate mean, min, max, and count using incremental computation.
        """
        file_path = self.clean_path(file_path)

        total_sum = None
        total_min = None
        total_max = None
        total_count = 0

        for chunk in self.stream_csv(file_path, chunk_size=chunk_size):
            if chunk is None:
                continue
            
            numeric_chunk = chunk.select_dtypes(include=[np.number])

            if numeric_chunk.empty:
                continue

            chunk_sum = numeric_chunk.sum()
            chunk_count = numeric_chunk.count()

            # Sum
            total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum

            # Count
            total_count += chunk_count.sum()

            # Min
            total_min = (
                numeric_chunk.min() if total_min is None 
                else np.minimum(total_min, numeric_chunk.min())
            )

            # Max
            total_max = (
                numeric_chunk.max() if total_max is None 
                else np.maximum(total_max, numeric_chunk.max())
            )

        if total_count == 0 or total_sum is None:
            return {"error": "No numeric data found"}

        return {
            "mean": (total_sum / total_count).round(4).tolist(),
            "min": pd.Series(total_min).tolist(),
            "max": pd.Series(total_max).tolist(),
            "count": int(total_count)
        }

    # ------------------------------------------------------------
    # 3️⃣ Detect anomalies in massive files
    # ------------------------------------------------------------
    def big_anomaly(self, file_path, chunk_size=50000, std_threshold=3):
        """
        Detect anomalies on huge datasets without loading all data in memory.
        """
        file_path = self.clean_path(file_path)

        anomalies_found = 0

        for chunk in self.stream_csv(file_path, chunk_size):
            if chunk is None:
                continue

            numeric_chunk = chunk.select_dtypes(include=[np.number])
            if numeric_chunk.empty:
                continue

            mean = numeric_chunk.mean()
            std = numeric_chunk.std()

            upper = mean + std_threshold * std
            lower = mean - std_threshold * std

            outliers = numeric_chunk[
                (numeric_chunk > upper) | (numeric_chunk < lower)
            ]

            # Count only meaningful anomalies
            if not outliers.dropna().empty:
                anomalies_found += 1

        return {
            "anomaly_chunks_found": anomalies_found,
            "details_available": anomalies_found > 0
        }

    # ------------------------------------------------------------
    # 4️⃣ Combined Big Data Summary Workflow
    # ------------------------------------------------------------
    def run(self, file_path):
        """
        Entry point for Big Data module.
        """
        file_path = self.clean_path(file_path)
        print(f"[BIGDATA] Processing huge dataset: {file_path}")

        stats = self.incremental_stats(file_path)
        anomalies = self.big_anomaly(file_path)

        return {
            "statistics": stats,
            "anomalies": anomalies
        }
