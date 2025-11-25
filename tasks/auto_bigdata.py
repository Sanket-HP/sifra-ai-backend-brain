# tasks/auto_bigdata.py

import numpy as np
import pandas as pd

class AutoBigData:
    """
    Lightweight Big Data Engine for SIFRA AI.
    Handles massive datasets using chunk streaming (memory-safe).
    """

    def __init__(self):
        print("[TASK] Auto BigData Engine Ready")

    # ------------------------------------------------------------
    # 1️⃣ Stream a large CSV file safely (chunk by chunk)
    # ------------------------------------------------------------
    def stream_csv(self, file_path, chunk_size=50000):
        """
        Reads large CSV files in chunks to avoid memory overflow.
        """
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            print(f"[BIGDATA ERROR] {str(e)}")
            return

    # ------------------------------------------------------------
    # 2️⃣ Compute statistics incrementally for huge data
    # ------------------------------------------------------------
    def incremental_stats(self, file_path, chunk_size=50000):
        """
        Calculate mean, min, max, and count using incremental computation.
        """
        total_sum = None
        total_min = None
        total_max = None
        total_count = 0

        for chunk in self.stream_csv(file_path, chunk_size=chunk_size):
            numeric_chunk = chunk.select_dtypes(include=[np.number])

            # Sum
            if total_sum is None:
                total_sum = numeric_chunk.sum()
            else:
                total_sum += numeric_chunk.sum()

            # Count
            total_count += numeric_chunk.count().sum()

            # Min
            if total_min is None:
                total_min = numeric_chunk.min()
            else:
                total_min = np.minimum(total_min, numeric_chunk.min())

            # Max
            if total_max is None:
                total_max = numeric_chunk.max()
            else:
                total_max = np.maximum(total_max, numeric_chunk.max())

        if total_count == 0:
            return {"error": "No numeric data found"}

        return {
            "mean": (total_sum / total_count).tolist(),
            "min": total_min.tolist(),
            "max": total_max.tolist(),
            "count": int(total_count)
        }

    # ------------------------------------------------------------
    # 3️⃣ Detect anomalies in massive files
    # ------------------------------------------------------------
    def big_anomaly(self, file_path, chunk_size=50000, std_threshold=3):
        """
        Detect anomalies on huge datasets without loading all data in memory.
        """
        anomalies = []

        for chunk in self.stream_csv(file_path, chunk_size):
            numeric_chunk = chunk.select_dtypes(include=[np.number])

            if numeric_chunk.empty:
                continue

            mean = numeric_chunk.mean()
            std = numeric_chunk.std()

            upper = mean + std_threshold * std
            lower = mean - std_threshold * std

            outliers = numeric_chunk[(numeric_chunk > upper) | (numeric_chunk < lower)]
            if not outliers.empty:
                anomalies.append(outliers)

        return {
            "anomaly_chunks_found": len(anomalies),
            "details_available": True if anomalies else False
        }

    # ------------------------------------------------------------
    # 4️⃣ Combined Big Data Summary Workflow
    # ------------------------------------------------------------
    def run(self, file_path):
        """
        Entry point for Big Data module.
        Returns:
        - Incremental stats
        - Big-anomaly detection
        """
        print("[BIGDATA] Processing huge dataset...")

        stats = self.incremental_stats(file_path)
        anomalies = self.big_anomaly(file_path)

        return {
            "statistics": stats,
            "anomalies": anomalies
        }
# Example usage:
# bigdata_engine = AutoBigData()