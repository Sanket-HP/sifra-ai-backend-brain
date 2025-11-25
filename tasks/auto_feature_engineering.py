# tasks/auto_feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


class AutoFeatureEngineering:
    """
    Automated Feature Engineering Engine for SIFRA AI.
    Enhances dataset structure for better ML performance.
    """

    def __init__(self):
        print("[TASK] Auto Feature Engineering Engine Ready")

    # ------------------------------------------------------------
    # Detect dtype
    # ------------------------------------------------------------
    def detect_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "categorical"

    # ------------------------------------------------------------
    # Extract polynomial features
    # ------------------------------------------------------------
    def generate_polynomials(self, df):
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 1:
            return df  # nothing to expand

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(numeric_df)

        poly_df = pd.DataFrame(
            poly_features,
            columns=poly.get_feature_names_out(numeric_df.columns)
        )

        poly_df = poly_df.reset_index(drop=True)
        df = df.reset_index(drop=True)

        return pd.concat([df, poly_df], axis=1)

    # ------------------------------------------------------------
    # Extract date features
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
    # MAIN ENGINE
    # ------------------------------------------------------------
    def run(self, dataset):
        """
        Accepts list/array → returns enhanced dataset + metadata.
        """

        # Convert dataset to DataFrame
        df = pd.DataFrame(dataset)

        # 🎯 FIX: Convert all column names to strings
        df.columns = df.columns.astype(str)

        # Try numeric or datetime conversion
        for col in df.columns:
            # Attempt numeric conversion
            df[col] = pd.to_numeric(df[col], errors="ignore")

            # Attempt datetime conversion
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                except:
                    pass

        # Detect types
        col_types = {col: self.detect_type(df[col]) for col in df.columns}

        # Fill NaNs properly (no warnings)
        df = df.ffill().bfill()

        # Extract date-based features
        df = self.extract_date_features(df)

        # One-hot encode categorical safely
        df = pd.get_dummies(df, drop_first=False)

        # Scale numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Add polynomial features
        df = self.generate_polynomials(df)

        # Remove constant columns
        df = df.loc[:, df.apply(pd.Series.nunique) > 1]

        # Final safe output
        return {
            "status": "success",
            "original_columns": list(col_types.keys()),
            "column_types": col_types,
            "final_shape": df.shape,
            "transformed_data": df.fillna(0).values.tolist()
        }
# -----------------------------------------------------------
# END OF FILE
# -----------------------------------------------------------