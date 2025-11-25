# config/settings.py

class Settings:
    """
    Global configuration for SIFRA AI.
    Modify values here without touching core logic.
    """

    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 5000

    # Forecasting default steps
    FORECAST_STEPS = 5

    # Logging
    ENABLE_LOGS = True

    # Preprocessor settings
    FILL_NAN_VALUE = 0
    DATE_CONVERSION_MODE = "timestamp"   # or 'ordinal'

    # Anomaly detection sensitivity
    ANOMALY_THRESHOLD_STD = 2.0

    # Trend calculation settings
    TREND_SMOOTHING = False

    # Insights
    TOP_INSIGHTS_LIMIT = 5
settings = Settings()