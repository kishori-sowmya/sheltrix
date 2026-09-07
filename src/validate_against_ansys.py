"""
Module 6 & 14: Validation of Top Recommended Designs against Physics & ANSYS
Checks surrogate vs Tier-1 physics model agreement, and reports ANSYS solver availability status.
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.rc_model import simulate_shelter
from src.physics.ansys_adapter import get_simulation_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_validation_report(top_n: int = 5) -> pd.DataFrame:
    pareto_path = os.path.join(BASE_DIR, "outputs", "pareto_front.csv")
    if not os.path.exists(pareto_path):
        from src.optimization.optimizer import run_optimization
        run_optimization()

    pareto_df = pd.read_csv(pareto_path)
    top_designs = pareto_df.head(top_n).reset_index(drop=True)
    ansys_engine = get_simulation_engine("ansys")

    records = []
    for i, row in top_designs.iterrows():
        design_dict = row.to_dict()
        surr_heating = float(row["heating_energy_required_kWh"])

        physics_res = simulate_shelter(design_dict, climate_scenario="winter")
        phys_heating = float(physics_res["heating_energy_required_kWh"])

        ansys_res = ansys_engine.run_simulation(design_dict, climate_scenario="winter")

        error_pct = abs(surr_heating - phys_heating) / max(phys_heating, 1e-4) * 100.0

        records.append({
            "design_rank": i + 1,
            "surrogate_heating_kWh": round(surr_heating, 2),
            "physics_rerun_heating_kWh": round(phys_heating, 2),
            "surrogate_vs_physics_error_pct": round(error_pct, 2),
            "ansys_status": ansys_res.get("message", "ANSYS solver unavailable"),
        })

    val_df = pd.DataFrame(records)
    out_path = os.path.join(BASE_DIR, "outputs", "validation_report.csv")
    val_df.to_csv(out_path, index=False)

    print(f"=== Stage 5 Validation Report (Top {top_n} Designs) ===")
    print(val_df.to_string(index=False))
    print(f"\nMean Surrogate-vs-Physics Error: {val_df['surrogate_vs_physics_error_pct'].mean():.2f}%")
    return val_df


if __name__ == "__main__":
    run_validation_report()
