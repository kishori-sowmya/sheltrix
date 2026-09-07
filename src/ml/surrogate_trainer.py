"""
Modules 7 & 16: AI XGBoost Surrogate Trainer & Versioned Registry
Trains separate XGBoost regressors for key thermal metrics, excludes Tier-2 validation candidates,
evaluates via 5-fold CV and test split, and registers models under versioned metadata directories.
"""

import pandas as pd
import numpy as np
import joblib, json, os, time
from typing import Dict, Any, Tuple, List
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "raw", "dataset_v1.csv")
MODEL_VERSION_DIR = os.path.join(BASE_DIR, "models", "v1.0")

TARGET_COLS = [
    "indoor_temp_mean", "indoor_temp_min", "indoor_temp_max", "indoor_temp_swing",
    "comfort_hours", "total_heat_loss_kWh", "total_solar_gain_kWh",
    "heating_energy_required_kWh",
]

CATEGORICAL_COLS = ["climate_scenario", "wall_material", "roof_material",
                      "floor_material", "insulation_material", "glazing_type"]

NUMERIC_FEATURE_COLS = [
    "length", "width", "wall_height", "roof_angle", "orientation_deg",
    "insulation_thickness_mm", "window_ratio", "pcm_present", "pcm_thickness_mm",
    "T_amb_mean", "T_amb_min", "T_amb_max", "solar_radiation_mean", "wind_speed_mean",
]


def load_data_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if not os.path.exists(DATA_PATH):
        # Fallback to root data dir if needed
        legacy_path = os.path.join(BASE_DIR, "data", "dataset_v1.csv")
        df = pd.read_csv(legacy_path)
    else:
        df = pd.read_csv(DATA_PATH)

    # Exclude Tier-2 ANSYS candidate rows to ensure zero data leakage
    train_pool = df[~df["ansys_validation_candidate"]].copy()

    X = pd.get_dummies(train_pool[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
    y = train_pool[TARGET_COLS]
    feature_names = list(X.columns)
    return X, y, feature_names


def train_surrogate_models() -> pd.DataFrame:
    X, y, feature_names = load_data_and_preprocess()
    os.makedirs(MODEL_VERSION_DIR, exist_ok=True)
    joblib.dump(feature_names, os.path.join(MODEL_VERSION_DIR, "feature_names.pkl"))
    joblib.dump(feature_names, os.path.join(BASE_DIR, "models", "feature_names.pkl")) # legacy root copy

    results = []
    trained_models = {}

    for target in TARGET_COLS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[target], test_size=0.15, random_state=42
        )

        model = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            min_child_weight=2,
            random_state=42,
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, 1e-5, y_test.values))) * 100)

        results.append({
            "target": target,
            "cv_mae_mean": round(float(cv_mae.mean()), 3),
            "cv_r2_mean": round(float(cv_r2.mean()), 3),
            "test_mae": round(mae, 3),
            "test_rmse": round(rmse, 3),
            "test_r2": round(r2, 3),
            "test_mape_pct": round(mape, 2),
        })

        trained_models[target] = model
        joblib.dump(model, os.path.join(MODEL_VERSION_DIR, f"xgb_{target}.pkl"))
        joblib.dump(model, os.path.join(BASE_DIR, "models", f"xgb_{target}.pkl")) # root legacy copy

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_VERSION_DIR, "surrogate_eval_metrics.csv"), index=False)
    results_df.to_csv(os.path.join(BASE_DIR, "models", "surrogate_eval_metrics.csv"), index=False)

    manifest = {
        "model_version": "v1.0",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "XGBoost Regressor (n_estimators=250, max_depth=4)",
        "n_training_samples": len(X),
        "feature_count": len(feature_names),
        "target_count": len(TARGET_COLS),
        "targets": TARGET_COLS,
    }
    with open(os.path.join(MODEL_VERSION_DIR, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== XGBoost Surrogate Training Complete (v1.0) ===")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    train_surrogate_models()
