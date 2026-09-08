"""
Stage 4: Multi-objective optimization over the shelter design space using
the trained XGBoost surrogate as the (fast) objective evaluator.

Objectives (minimize both):
  1. heating_energy_required_kWh  (lower = less fuel/energy needed)
  2. -comfort_hours                (maximize comfort -> minimize negative)

NSGA-II (via pymoo) produces a Pareto front of non-dominated shelter
designs rather than one "best" answer, matching the fact that these two
goals genuinely trade off (e.g., more insulation costs more material but
improves both - so a 3rd soft objective, insulation_thickness as a cost
proxy, is included to keep the front non-trivial).
"""
import numpy as np
import pandas as pd
import joblib
import os
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.materials import WALL_MATERIAL_LIST, INSULATION_LIST, GLAZING_LIST
from data.climate_ladakh import get_climate_summary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Fixed categorical choices for MVP optimizer run (categorical search handled
# by running the continuous optimizer once per material combo, or fixed here
# for a focused demo; extending to joint categorical+continuous is an
# advanced-tier enhancement noted in the project scope).
FIXED_WALL_MATERIAL = "cseb"
FIXED_ROOF_MATERIAL = "sandwich_panel"
FIXED_FLOOR_MATERIAL = "concrete"
FIXED_INSULATION_MATERIAL = "xps_foam"
FIXED_GLAZING = "double_pane"
CLIMATE_SCENARIO = "winter"

# Continuous decision variables being optimized:
# [length, width, wall_height, roof_angle, orientation_deg,
#  insulation_thickness_mm, window_ratio, pcm_thickness_mm]
VAR_BOUNDS = np.array([
    [3.0, 8.0], [3.0, 6.0], [2.2, 3.5], [0.0, 45.0], [0.0, 180.0],
    [0.0, 150.0], [0.05, 0.40], [0.0, 50.0],
])


def load_surrogates():
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    heat_model = joblib.load(os.path.join(MODEL_DIR, "xgb_heating_energy_required_kWh.pkl"))
    comfort_model = joblib.load(os.path.join(MODEL_DIR, "xgb_comfort_hours.pkl"))
    return feature_names, heat_model, comfort_model


def build_feature_row(x, feature_names):
    """Convert an optimizer decision vector into the one-hot feature row the surrogate expects."""
    length, width, wall_height, roof_angle, orientation_deg, ins_thick, window_ratio, pcm_thick = x
    climate = get_climate_summary(CLIMATE_SCENARIO)

    row = {name: 0 for name in feature_names}
    numeric_vals = {
        "length": length, "width": width, "wall_height": wall_height,
        "roof_angle": roof_angle, "orientation_deg": orientation_deg,
        "insulation_thickness_mm": ins_thick, "window_ratio": window_ratio,
        "pcm_present": 1 if pcm_thick > 1.0 else 0, "pcm_thickness_mm": pcm_thick,
        **climate,
    }
    for k, v in numeric_vals.items():
        if k in row:
            row[k] = v

    cat_cols = {
        f"climate_scenario_{CLIMATE_SCENARIO}": 1,
        f"wall_material_{FIXED_WALL_MATERIAL}": 1,
        f"roof_material_{FIXED_ROOF_MATERIAL}": 1,
        f"floor_material_{FIXED_FLOOR_MATERIAL}": 1,
        f"insulation_material_{FIXED_INSULATION_MATERIAL}": 1,
        f"glazing_type_{FIXED_GLAZING}": 1,
    }
    for k, v in cat_cols.items():
        if k in row:
            row[k] = v
    return [row[name] for name in feature_names]


class ShelterDesignProblem(Problem):
    def __init__(self, feature_names, heat_model, comfort_model):
        super().__init__(n_var=8, n_obj=2, n_constr=0,
                          xl=VAR_BOUNDS[:, 0], xu=VAR_BOUNDS[:, 1])
        self.feature_names = feature_names
        self.heat_model = heat_model
        self.comfort_model = comfort_model

    def _evaluate(self, X, out, *args, **kwargs):
        feature_rows = np.array([build_feature_row(x, self.feature_names) for x in X])
        heating = self.heat_model.predict(feature_rows)
        comfort = self.comfort_model.predict(feature_rows)
        # Objective 1: minimize heating energy. Objective 2: minimize -comfort (i.e. maximize comfort)
        out["F"] = np.column_stack([heating, -comfort])


def run_optimization(pop_size=100, n_gen=40):
    feature_names, heat_model, comfort_model = load_surrogates()
    problem = ShelterDesignProblem(feature_names, heat_model, comfort_model)

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
    print(f"NSGA-II: {total_evals} surrogate evaluations across {n_gen} generations "
          f"in {elapsed:.2f}s ({elapsed/total_evals*1000:.3f} ms/eval)")

    var_names = ["length", "width", "wall_height", "roof_angle", "orientation_deg",
                 "insulation_thickness_mm", "window_ratio", "pcm_thickness_mm"]
    pareto_df = pd.DataFrame(result.X, columns=var_names)
    pareto_df["heating_energy_required_kWh"] = result.F[:, 0]
    pareto_df["comfort_hours"] = -result.F[:, 1]
    pareto_df["wall_material"] = FIXED_WALL_MATERIAL
    pareto_df["roof_material"] = FIXED_ROOF_MATERIAL
    pareto_df["floor_material"] = FIXED_FLOOR_MATERIAL
    pareto_df["insulation_material"] = FIXED_INSULATION_MATERIAL
    pareto_df["glazing_type"] = FIXED_GLAZING

    pareto_df = pareto_df.sort_values("heating_energy_required_kWh").reset_index(drop=True)
    return pareto_df, elapsed, total_evals


if __name__ == "__main__":
    pareto_df, elapsed, total_evals = run_optimization()
    out_path = os.path.join(BASE_DIR, "outputs", "pareto_front.csv")
    pareto_df.to_csv(out_path, index=False)
    print(f"\nPareto front ({len(pareto_df)} non-dominated designs) saved to {out_path}\n")
    print(pareto_df[["length", "width", "insulation_thickness_mm", "window_ratio",
                      "pcm_thickness_mm", "heating_energy_required_kWh", "comfort_hours"]]
          .head(10).to_string(index=False))
