"""
Modules 9 & 10: Multi-Objective NSGA-II Optimizer & Engineering Feasibility Filter
Performs joint continuous and material categorical optimization using trained XGBoost surrogates.
Returns non-dominated Pareto designs and candidate highlights (Best Comfort, Best Energy, Best Cost, Balanced).
"""

import numpy as np
import pandas as pd
import joblib, os, time
from typing import Dict, Any, Tuple, List
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_modules.material_db import MATERIAL_DATABASE, WALL_MATERIAL_LIST, INSULATION_LIST, GLAZING_LIST
from src.data_modules.climate_manager import ClimateManager
from src.ml.quality_control import SurrogateQualityControl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Continuous Decision Variables:
# [0: length, 1: width, 2: wall_height, 3: roof_angle, 4: orientation_deg,
#  5: insulation_thickness_mm, 6: window_ratio, 7: pcm_thickness_mm]
VAR_BOUNDS = np.array([
    [3.0, 8.0],   # length (m)
    [3.0, 6.0],   # width (m)
    [2.2, 3.5],   # wall_height (m)
    [0.0, 45.0],  # roof_angle (deg)
    [0.0, 180.0], # orientation_deg (0=South)
    [0.0, 150.0], # insulation_thickness_mm
    [0.05, 0.40], # window_ratio
    [0.0, 50.0],  # pcm_thickness_mm
])


def load_surrogate_models() -> Tuple[List[str], Any, Any, Any]:
    feat_path = os.path.join(MODEL_DIR, "v1.0", "feature_names.pkl")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    feature_names = joblib.load(feat_path)
    heat_path = os.path.join(MODEL_DIR, "v1.0", "xgb_heating_energy_required_kWh.pkl")
    if not os.path.exists(heat_path):
        heat_path = os.path.join(MODEL_DIR, "xgb_heating_energy_required_kWh.pkl")

    comf_path = os.path.join(MODEL_DIR, "v1.0", "xgb_comfort_hours.pkl")
    if not os.path.exists(comf_path):
        comf_path = os.path.join(MODEL_DIR, "xgb_comfort_hours.pkl")

    heat_loss_path = os.path.join(MODEL_DIR, "v1.0", "xgb_total_heat_loss_kWh.pkl")
    if not os.path.exists(heat_loss_path):
        heat_loss_path = os.path.join(MODEL_DIR, "xgb_total_heat_loss_kWh.pkl")

    heat_model = joblib.load(heat_path)
    comfort_model = joblib.load(comf_path)
    heat_loss_model = joblib.load(heat_loss_path)
    return feature_names, heat_model, comfort_model, heat_loss_model


def build_surrogate_feature_vector(x: np.ndarray, feature_names: List[str], climate_scenario: str, materials_cfg: Dict[str, str]) -> List[float]:
    """Convert optimizer continuous decision vector into surrogate ML one-hot feature vector."""
    L, W, H, roof_angle, orientation, ins_thick, window_ratio, pcm_thick = x

    cm = ClimateManager()
    climate_summary = cm.get_summary(climate_scenario)

    row = {name: 0.0 for name in feature_names}
    num_vals = {
        "length": float(L),
        "width": float(W),
        "wall_height": float(H),
        "roof_angle": float(roof_angle),
        "orientation_deg": float(orientation),
        "insulation_thickness_mm": float(ins_thick),
        "window_ratio": float(window_ratio),
        "pcm_present": 1.0 if pcm_thick > 1.0 else 0.0,
        "pcm_thickness_mm": float(pcm_thick),
        **climate_summary,
    }
    for k, v in num_vals.items():
        if k in row:
            row[k] = v

    cat_dummies = [
        f"climate_scenario_{climate_scenario}",
        f"wall_material_{materials_cfg.get('wall_material', 'cseb')}",
        f"roof_material_{materials_cfg.get('roof_material', 'sandwich_panel')}",
        f"floor_material_{materials_cfg.get('floor_material', 'concrete')}",
        f"insulation_material_{materials_cfg.get('insulation_material', 'xps_foam')}",
        f"glazing_type_{materials_cfg.get('glazing_type', 'double_pane')}",
    ]
    for cat in cat_dummies:
        if cat in row:
            row[cat] = 1.0

    return [row[name] for name in feature_names]


class FeasibilityChecker:
    """Module 10: Engineering Feasibility Layer validating physical design constraints."""

    @staticmethod
    def is_feasible(design: Dict[str, Any], max_footprint: float = 48.0) -> Tuple[bool, List[str]]:
        reasons = []
        L, W = design["length"], design["width"]
        footprint = L * W
        aspect_ratio = L / max(W, 0.1)

        if footprint > max_footprint:
            reasons.append(f"Footprint {footprint:.1f}m2 exceeds maximum allowed {max_footprint}m2.")
        if aspect_ratio < 0.8 or aspect_ratio > 3.0:
            reasons.append(f"Aspect ratio {aspect_ratio:.2f} is structurally impractical (must be 0.8 to 3.0).")
        if design["window_ratio"] > 0.40:
            reasons.append("Window ratio exceeds maximum 40% thermal envelope limit.")
        if design["insulation_thickness_mm"] < 10.0:
            reasons.append("Insulation thickness below minimum 10mm cold-climate threshold.")

        return len(reasons) == 0, reasons


class ShelterDesignProblem(Problem):

    def __init__(self, feature_names: List[str], heat_model: Any, comfort_model: Any, climate_scenario: str, materials_cfg: Dict[str, str]):
        super().__init__(
            n_var=8,
            n_obj=3, # 1: heating kWh, 2: -comfort_hours, 3: cost proxy
            n_constr=1, # aspect ratio constraint
            xl=VAR_BOUNDS[:, 0],
            xu=VAR_BOUNDS[:, 1],
        )
        self.feature_names = feature_names
        self.heat_model = heat_model
        self.comfort_model = comfort_model
        self.climate_scenario = climate_scenario
        self.materials_cfg = materials_cfg
        self.qc = SurrogateQualityControl()

    def _evaluate(self, X, out, *args, **kwargs):
        feature_matrix = [
            build_surrogate_feature_vector(x, self.feature_names, self.climate_scenario, self.materials_cfg)
            for x in X
        ]
        heating = self.heat_model.predict(feature_matrix)
        comfort = self.comfort_model.predict(feature_matrix)

        # Objective 3: Construction cost proxy (INR/m2 envelope area)
        cost_proxy = np.zeros(len(X))
        for i, x in enumerate(X):
            L, W, H, _, _, ins_t, w_ratio, pcm_t = x
            area = 2 * (L + W) * H + L * W
            ins_cost = (ins_t / 50.0) * 1250.0 * area
            pcm_cost = (pcm_t / 10.0) * 2000.0 * (L * W)
            cost_proxy[i] = ins_cost + pcm_cost + (area * 3000.0)

        # F1: Min heating, F2: Max comfort (-comfort), F3: Min cost
        out["F"] = np.column_stack([heating, -comfort, cost_proxy])

        # Constraint: Aspect ratio L/W <= 2.8 (g <= 0)
        g1 = (X[:, 0] / X[:, 1]) - 2.8
        out["G"] = g1


def run_optimization(
    climate_scenario: str = "winter",
    materials_cfg: Dict[str, str] = None,
    pop_size: int = 100,
    n_gen: int = 40,
    max_footprint: float = 48.0,
) -> Tuple[pd.DataFrame, float, int, Dict[str, Dict[str, Any]]]:

    if materials_cfg is None:
        materials_cfg = {
            "wall_material": "cseb",
            "roof_material": "sandwich_panel",
            "floor_material": "concrete",
            "insulation_material": "xps_foam",
            "glazing_type": "double_pane",
        }

    feature_names, heat_model, comfort_model, _ = load_surrogate_models()
    problem = ShelterDesignProblem(feature_names, heat_model, comfort_model, climate_scenario, materials_cfg)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    t0 = time.time()
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=False)
    elapsed = time.time() - t0
    total_evals = pop_size * n_gen

    var_names = ["length", "width", "wall_height", "roof_angle", "orientation_deg",
                 "insulation_thickness_mm", "window_ratio", "pcm_thickness_mm"]

    pareto_df = pd.DataFrame(result.X, columns=var_names)
    pareto_df["heating_energy_required_kWh"] = np.maximum(0.0, result.F[:, 0])
    pareto_df["comfort_hours"] = np.clip(-result.F[:, 1], 0.0, 24.0)
    pareto_df["estimated_cost_inr"] = result.F[:, 2]
    pareto_df["estimated_cost_usd"] = result.F[:, 2] # Alias for legacy compatibility

    for k, v in materials_cfg.items():
        pareto_df[k] = v

    pareto_df["footprint_m2"] = pareto_df["length"] * pareto_df["width"]

    # Filter engineering feasibility
    feasible_mask = []
    for _, row in pareto_df.iterrows():
        is_feas, _ = FeasibilityChecker.is_feasible(row.to_dict(), max_footprint=max_footprint)
        feasible_mask.append(is_feas)

    pareto_df["is_feasible"] = feasible_mask
    pareto_df = pareto_df[pareto_df["is_feasible"]].sort_values("heating_energy_required_kWh").reset_index(drop=True)

    # Highlight top candidates
    candidates = {
        "Best Energy": pareto_df.sort_values("heating_energy_required_kWh").iloc[0].to_dict(),
        "Best Comfort": pareto_df.sort_values("comfort_hours", ascending=False).iloc[0].to_dict(),
        "Best Cost": pareto_df.sort_values("estimated_cost_inr").iloc[0].to_dict(),
        "Balanced": pareto_df.iloc[len(pareto_df) // 2].to_dict(),
    }

    # Save output
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    pareto_df.to_csv(os.path.join(out_dir, "pareto_front.csv"), index=False)

    return pareto_df, elapsed, total_evals, candidates


if __name__ == "__main__":
    df, elapsed, total_evals, candidates = run_optimization()
    print(f"NSGA-II Optimization completed: {total_evals} evals in {elapsed:.2f}s")
    print("Pareto Candidates Found:")
    for k, c in candidates.items():
        print(f"  {k}: {c['heating_energy_required_kWh']:.1f} kWh/day, {c['comfort_hours']:.1f} comfort hrs")
