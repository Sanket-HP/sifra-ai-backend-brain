# tasks/auto_visualize.py

import numpy as np

class AutoVisualize:
    """
    Lightweight Auto Visualization engine for SIFRA AI.
    It generates visualization instructions instead of real charts.
    This keeps the engine offline and lightweight.
    """

    def __init__(self):
        print("[TASK] Auto Visualization Module Ready")

    def detect_chart_type(self, arr):
        """
        Decide which chart is best:
        - Line chart: numeric sequences
        - Bar chart: categorical values
        - Scatter: paired numeric data
        """
        arr = np.array(arr)

        # 2D data = scatter plot
        if len(arr.shape) == 2 and arr.shape[1] == 2:
            return "scatter"

        # 1D numeric = line chart
        if np.issubdtype(arr.dtype, np.number):
            return "line"

        # fallback
        return "bar"

    def create_visual_plan(self, dataset):
        """
        Create a visualization plan:
        - chart type
        - x-values
        - y-values
        """
        arr = np.array(dataset)

        chart_type = self.detect_chart_type(arr)

        plan = {
            "chart_type": chart_type,
            "description": f"Recommended chart: {chart_type}",
        }

        # scatter plots
        if chart_type == "scatter" and arr.shape[1] == 2:
            plan["x"] = arr[:, 0].tolist()
            plan["y"] = arr[:, 1].tolist()
            return plan

        # line / bar chart
        if len(arr.shape) == 1:
            plan["x"] = list(range(len(arr)))
            plan["y"] = arr.tolist()
        else:
            plan["x"] = list(range(arr.shape[0]))
            plan["y"] = arr[:, 0].tolist()

        return plan

    def run(self, dataset):
        """
        Main entry for visualization engine.
        """
        try:
            plan = self.create_visual_plan(dataset)
            return {
                "status": "success",
                "visual_plan": plan
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
