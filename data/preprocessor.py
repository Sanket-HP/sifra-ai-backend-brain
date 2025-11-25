# data/preprocessor.py

import numpy as np
import pandas as pd

class Preprocessor:
    """
    Cleans and converts mixed datasets (text, dates, NaN) into
    numeric-only arrays for SIFRA AI analysis.
    """

    def __init__(self):
        print("[DATA] Preprocessor Ready")

    def clean(self, data):
        """
        Main entry for cleaning the dataset.
        Accepts pandas.DataFrame or numpy array.
        Returns clean numeric numpy array.
        """

        # Convert raw numpy → pandas
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data

        print("[PREPROCESSOR] Initial shape:", df.shape)

        # 1. Remove completely empty rows
        df = df.dropna(how='all')
        print("[PREPROCESSOR] Removed empty rows. New shape:", df.shape)

        # 2. Remove columns that are entirely NaN or empty
        df = df.dropna(how='all', axis=1)
        print("[PREPROCESSOR] Removed empty columns. New shape:", df.shape)

        # 3. Convert dates to numeric timestamps
        df = df.apply(self._convert_dates)
        print("[PREPROCESSOR] Converted dates.")

        # 4. Convert text columns to numeric category codes
        df = df.apply(self._text_to_numeric)
        print("[PREPROCESSOR] Converted text to numeric.")

        # 5. Replace remaining NaN with 0
        df = df.fillna(0)

        # 6. Final numeric conversion
        numeric_data = df.values.astype(float)

        print("[PREPROCESSOR] Final cleaned shape:", numeric_data.shape)

        return numeric_data

    def _convert_dates(self, col):
        """
        Converts datetime values into numeric timestamp integers.
        """
        if pd.api.types.is_datetime64_any_dtype(col):
            return col.astype("int64") // 10**9  # convert to seconds
        return col

    def _text_to_numeric(self, col):
        """
        Converts any text column to numeric category IDs.
        Example: Kolhapur → 1, Pune → 2, Sangali → 3
        """
        if col.dtype == object:
            try:
                return col.astype(float)  # if numeric text, convert directly
            except:
                # Convert strings to category codes
                return col.astype("category").cat.codes
        return col
# Example usage:
# preprocessor = Preprocessor()
# clean_data = preprocessor.clean(your_dataset)