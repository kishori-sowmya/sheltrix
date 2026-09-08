"""
Module 5: Latin Hypercube Sampling & Simulation Dataset Pipeline
Generates reproducible physics simulation datasets across parameter space and climate scenarios.
Saves structured outputs to datasets/raw/, datasets/processed/, and datasets/metadata/.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml, json, os, time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.physics.rc_model import simulate_shelter
from src.data_modules.climate_manager import ClimateManager
from src.data_modules.material_db import WALL_MATERIAL_LIST, INSULATION_LIST, GLAZING_LIST

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "design_parameters.yaml")

with open(CONFIG_PATH, "r") as f:
    PARAM_CONFIG = yaml.safe_load(f)["parameters"]

CONT_KEYS = [
    "length", "width", "wall_height", "roof_angle", "orientation_deg",
    "insulation_thickness_mm", "window_ratio", "pcm_thickness_mm"
]


def sample_parameter_space(n_samples: int = 300, seed: int = 42) -> list:
    """Generate n_samples using Latin Hypercube Sampling (LHS) across design bounds."""
    sampler = qmc.LatinHypercube(d=len(CONT_KEYS), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lower = np.array([PARAM_CONFIG[k]["min"] for k in CONT_KEYS])
    upper = np.array([PARAM_CONFIG[k]["max"] for k in CONT_KEYS])
    scaled = qmc.scale(unit_samples, lower, upper)

    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_samples):
        row = dict(zip(CONT_KEYS, scaled[i]))
        row["wall_material"] = str(rng.choice(WALL_MATERIAL_LIST))
        row["roof_material"] = str(rng.choice(WALL_MATERIAL_LIST))
        row["floor_material"] = str(rng.choice(["concrete", "brick", "cseb"]))
        row["insulation_material"] = str(rng.choice(INSULATION_LIST))
        row["glazing_type"] = str(rng.choice(GLAZING_LIST))
        row["pcm_present"] = int(rng.random() < 0.4)
        if row["pcm_present"] == 0:
            row["pcm_thickness_mm"] = 0.0
        row["sample_id"] = i
        configs.append(row)
    return configs


def build_and_save_dataset(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate complete physics dataset across climate scenarios and store with provenance."""
    cm = ClimateManager()
    climates = ["winter", "summer"]
    configs = sample_parameter_space(n_samples=n_samples, seed=seed)

    rows = []
    t0 = time.time()

    for scenario in climates:
        climate_summary = cm.get_summary(scenario)
        climate_prof = cm.get_profile(scenario)

        for cfg in configs:
            sim_start = time.time()
            res = simulate_shelter(cfg, climate_dict=climate_prof)
            sim_time = time.time() - sim_start

            row = {
                "sample_id": f"{scenario}_{cfg['sample_id']}",
                "climate_scenario": scenario,
                **{k: cfg[k] for k in CONT_KEYS},
                "wall_material": cfg["wall_material"],
                "roof_material": cfg["roof_material"],
                "floor_material": cfg["floor_material"],
                "insulation_material": cfg["insulation_material"],
                "glazing_type": cfg["glazing_type"],
                "pcm_present": cfg["pcm_present"],
                **climate_summary,
                "indoor_temp_mean": res["indoor_temp_mean"],
                "indoor_temp_min": res["indoor_temp_min"],
                "indoor_temp_max": res["indoor_temp_max"],
                "indoor_temp_swing": res["indoor_temp_swing"],
                "comfort_hours": res["comfort_hours"],
                "total_heat_loss_kWh": res["total_heat_loss_kWh"],
                "total_solar_gain_kWh": res["total_solar_gain_kWh"],
                "ventilation_loss_kWh": res["ventilation_loss_kWh"],
                "heating_energy_required_kWh": res["heating_energy_required_kWh"],
                "fidelity_tier": "tier1_rc_physics",
                "simulation_time_sec": sim_time,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Stratify 30 candidates for Tier-2 ANSYS validation
    winter_df = df[df["climate_scenario"] == "winter"].copy()
    winter_sorted = winter_df.sort_values("insulation_thickness_mm")
    stratified_idx = np.linspace(0, len(winter_sorted) - 1, 30, dtype=int)
    tier2_ids = set(winter_sorted.iloc[stratified_idx]["sample_id"])
    df["ansys_validation_candidate"] = df["sample_id"].isin(tier2_ids)

    # Save raw, processed, metadata
    raw_dir = os.path.join(BASE_DIR, "datasets", "raw")
    meta_dir = os.path.join(BASE_DIR, "datasets", "metadata")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, "dataset_v1.csv")
    df.to_csv(csv_path, index=False)

    # Legacy copy for root compatibility
    legacy_csv = os.path.join(BASE_DIR, "data", "dataset_v1.csv")
    os.makedirs(os.path.dirname(legacy_csv), exist_ok=True)
    df.to_csv(legacy_csv, index=False)

    metadata = {
        "dataset_version": "v1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples_per_scenario": n_samples,
        "total_rows": len(df),
        "random_seed": seed,
        "climate_scenarios": climates,
        "physics_engine": "Tier-1 Lumped Capacitance RC Network with 3-day spinup",
        "filepath": csv_path,
    }
    with open(os.path.join(meta_dir, "dataset_v1_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(df)} simulation dataset rows in {time.time()-t0:.2f}s")
    return df


if __name__ == "__main__":
    df = build_and_save_dataset()
