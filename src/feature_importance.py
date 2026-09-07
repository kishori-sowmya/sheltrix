"""Stage 3.5: Feature importance sanity check for key surrogate outputs."""
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

KEY_TARGETS = ["indoor_temp_mean", "heating_energy_required_kWh", "comfort_hours", "total_solar_gain_kWh"]

def print_importance():
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    for target in KEY_TARGETS:
        model = joblib.load(os.path.join(MODEL_DIR, f"xgb_{target}.pkl"))
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(8)
        print(f"\n--- Top features for {target} ---")
        print(imp_df.to_string(index=False))

if __name__ == "__main__":
    print_importance()
