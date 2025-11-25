# tasks/auto_analyze.py

from core.sifra_core import SifraCore
from data.preprocessor import Preprocessor

class AutoAnalyze:
    """
    Autonomous analysis task for SIFRA AI.
    """

    def __init__(self):
        self.core = SifraCore()
        self.preprocessor = Preprocessor()
        print("[TASK] Auto Analyze Module Ready")

    def run(self, dataset):
        print("\n[AUTO ANALYZE] Running Autonomous Analysis...")

        # Clean dataset first
        clean_data = self.preprocessor.clean(dataset)

        # Run SIFRA Core
        result = self.core.run("analyze", clean_data)

        return {
            "task": "auto_analyze",
            "intent_used": result["intent_vector"],
            "analysis_result": result["analysis_result"],
            "message": "Analysis completed successfully"
        }
# Example usage:
# auto_analyze = AutoAnalyze()
# analysis_output = auto_analyze.run(your_dataset)