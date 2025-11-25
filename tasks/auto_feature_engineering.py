# tasks/auto_feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

class AutoFeatureEngineering:
    """
    Automated Feature Engineering Engine for SIFRA AI.
    Enhances dataset structure for better ML performance.
    """

    def __init__(self):
        print("[TASK] Auto Feature Engineering Engine Ready")

    # ------------------------------------------------------------
    # Helper: Detect column type
    # ------------------------------------------------------------
    def detect_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "categorical"

    # ------------------------------------------------------------
    # Helper: Generate polynomial features
    # ------------------------------------------------------------
    def generate_polynomials(self, df):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 1:
            return df

        poly_features = poly.fit_transform(numeric_df)
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_df.columns))

        return pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------
    # Helper: Extract features from datetime
    # ------------------------------------------------------------
    def extract_date_features(self, df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
        return df

    # ------------------------------------------------------------
    # MAIN FUNCTION
    # ------------------------------------------------------------
    def run(self, dataset):
        """
        Accepts list/array → returns enhanced dataset + metadata.
        """

        df = pd.DataFrame(dataset)

        # Convert to numeric or datetime where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        # Detect column types
        col_types = {col: self.detect_type(df[col]) for col in df.columns}

        # Handle missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Extract date features
        df = self.extract_date_features(df)

        # One-hot encode categorical columns
        df = pd.get_dummies(df, drop_first=True)

        # Scaling (MinMax)
        scaler = MinMaxScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Add polynomial features
        df = self.generate_polynomials(df)

        # Remove constant columns
        df = df.loc[:, df.apply(pd.Series.nunique) > 1]

        return {
            "status": "success",
            "original_columns": list(col_types.keys()),
            "column_types": col_types,
            "final_shape": df.shape,
            "transformed_data": df.fillna(0).values.tolist()
        }
