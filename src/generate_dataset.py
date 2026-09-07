"""
Stage 2 pipeline: generate the full Tier-1 (RC-model) dataset, plus flag
a stratified subset as Tier-2 candidates (meant for real ANSYS runs).
"""
import pandas as pd
import numpy as np
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sampler import generate_samples
from src.thermal_model import simulate_shelter
from data.climate_ladakh import get_climate_summary

N_SAMPLES = 300
CLIMATE_SCENARIOS_TO_RUN = ["winter", "summer"]
N_TIER2_STRATIFIED = 30


def build_dataset():
    samples = generate_samples(n_samples=N_SAMPLES)
    rows = []
    t0 = time.time()

    for scenario in CLIMATE_SCENARIOS_TO_RUN:
        climate_feats = get_climate_summary(scenario)
        for cfg in samples:
            sim_start = time.time()
            outputs = simulate_shelter(cfg, climate_scenario=scenario)
            sim_time = time.time() - sim_start

            row = {
                "sample_id": f"{scenario}_{cfg['sample_id']}",
                "climate_scenario": scenario,
                **{k: cfg[k] for k in [
                    "length", "width", "wall_height", "roof_angle", "orientation_deg",
                    "wall_material", "roof_material", "floor_material",
                    "insulation_material", "insulation_thickness_mm", "window_ratio",
                    "pcm_present", "pcm_thickness_mm", "glazing_type",
                ]},
                **climate_feats,
                **outputs,
                "fidelity_tier": "tier1_rc",
                "simulation_time_sec": sim_time,
            }
            rows.append(row)

    total_time = time.time() - t0
    df = pd.DataFrame(rows)

    # --- Data quality checks ---
    n_before = len(df)
    df = df[(df["indoor_temp_mean"] > -40) & (df["indoor_temp_mean"] < 60)]
    n_after = len(df)
    if n_before != n_after:
        print(f"WARNING: dropped {n_before - n_after} rows failing sanity bounds")

    # --- Stratify a Tier-2 (ANSYS-candidate) subset ---
    # Pick a spread across insulation thickness and window ratio to cover the design space
    winter_df = df[df["climate_scenario"] == "winter"].copy()
    winter_df_sorted = winter_df.sort_values("insulation_thickness_mm")
    stratified_idx = np.linspace(0, len(winter_df_sorted) - 1, N_TIER2_STRATIFIED, dtype=int)
    tier2_candidates = winter_df_sorted.iloc[stratified_idx]["sample_id"].tolist()
    df["ansys_validation_candidate"] = df["sample_id"].isin(tier2_candidates)

    print(f"Generated {len(df)} rows across {len(CLIMATE_SCENARIOS_TO_RUN)} climate scenarios "
          f"in {total_time:.2f}s (avg {total_time/len(df)*1000:.2f} ms/sample)")
    print(f"Flagged {df['ansys_validation_candidate'].sum()} rows as ANSYS validation candidates (Tier-2)")

    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_v1.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
    print(df.describe())
