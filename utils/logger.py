# utils/logger.py

import logging
import os

class SifraLogger:
    """
    Central logging utility for SIFRA AI.
    Logs to console + file: logs/sifra.log
    """

    def __init__(self, name="SIFRA_AI"):
        # Ensure logs folder exists
        os.makedirs("logs", exist_ok=True)

        log_file = "logs/sifra.log"

        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Avoid double logging
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
# Singleton instance
sifra_logger = SifraLogger()