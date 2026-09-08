"""
Module 8: Surrogate Quality Control & Domain Bounds Validator
Checks model availability, performance metric thresholds, and validates input feature ranges
prior to allowing optimization or batch surrogate predictions.
"""

import os, json
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Tuple, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "design_parameters.yaml")
METRICS_PATH = os.path.join(BASE_DIR, "models", "v1.0", "surrogate_eval_metrics.csv")

with open(CONFIG_PATH, "r") as f:
    PARAM_CONFIG = yaml.safe_load(f)["parameters"]


class SurrogateQualityControl:

    def __init__(self, min_r2_threshold: float = 0.80):
        self.min_r2_threshold = min_r2_threshold

    def is_surrogate_ready(self) -> Tuple[bool, str]:
        """Verify that surrogate model artifacts exist and meet evaluation quality thresholds."""
        if not os.path.exists(METRICS_PATH):
            # Check root legacy fallback
            root_metrics = os.path.join(BASE_DIR, "models", "surrogate_eval_metrics.csv")
            if not os.path.exists(root_metrics):
                return False, "Surrogate model metrics not found. Please train models first."

        metrics_df = pd.read_csv(METRICS_PATH if os.path.exists(METRICS_PATH) else root_metrics)
        key_targets = ["heating_energy_required_kWh", "comfort_hours", "indoor_temp_mean"]
        sub_df = metrics_df[metrics_df["target"].isin(key_targets)]

        low_perf = sub_df[sub_df["test_r2"] < self.min_r2_threshold]
        if not low_perf.empty:
            return False, f"Surrogate R2 below threshold {self.min_r2_threshold} for targets: {low_perf['target'].tolist()}"

        return True, "Surrogate models verified and passed quality control."

    def validate_input_bounds(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input design parameters against continuous training bounds."""
        warnings = []
        for param, config in PARAM_CONFIG.items():
            if param in params:
                val = params[param]
                p_min, p_max = config["min"], config["max"]
                if val < p_min or val > p_max:
                    warnings.append(f"Parameter '{param}'={val} is outside training range [{p_min}, {p_max}]. Prediction may extrapolate.")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def sanitize_prediction(self, raw_pred: float, target: str) -> float:
        """Sanitize raw ML outputs against physical lower/upper bounds."""
        if target == "comfort_hours":
            return float(np.clip(raw_pred, 0.0, 24.0))
        elif target in ["heating_energy_required_kWh", "total_heat_loss_kWh", "total_solar_gain_kWh", "indoor_temp_swing"]:
            return float(np.maximum(0.0, raw_pred))
        elif target in ["indoor_temp_mean", "indoor_temp_min", "indoor_temp_max"]:
            return float(np.clip(raw_pred, -50.0, 60.0))
        return float(raw_pred)


if __name__ == "__main__":
    qc = SurrogateQualityControl()
    ready, msg = qc.is_surrogate_ready()
    print("Quality Control Check:", ready, "|", msg)
    valid, warn = qc.validate_input_bounds({"length": 12.0, "width": 4.0})
    print("Bounds Validation:", valid, "| Warnings:", warn)
