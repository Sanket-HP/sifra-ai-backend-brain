# data/dataset_loader.py

import pandas as pd
import numpy as np

class DatasetLoader:
    """
    Loads datasets from multiple formats for SIFRA AI.
    Supports CSV, Excel, JSON, and raw lists.
    """

    def __init__(self):
        print("[DATA] Dataset Loader Ready")

    def load_csv(self, path):
        print(f"[DATA] Loading CSV: {path}")
        return pd.read_csv(path).values

    def load_excel(self, path):
        print(f"[DATA] Loading Excel: {path}")
        return pd.read_excel(path).values

    def load_json(self, path):
        print(f"[DATA] Loading JSON: {path}")
        df = pd.read_json(path)
        return df.values

    def load_raw(self, data):
        """
        Accepts raw Python lists or NumPy arrays.
        """
        print("[DATA] Loading Raw Dataset")
        return np.array(data)
# Example usage:
# loader = DatasetLoader()
# dataset = loader.load_csv('data.csv') or loader.load_raw([[1,2,3],[4,5,6]])