"""
Stage 3: Train one XGBoost regressor per output target (Option A from the
design doc: separate models per output for robustness/explainability).
Reports MAE / RMSE / R2 via 5-fold CV and on a held-out test split.
Tier-2 (ANSYS) rows are EXCLUDED from training entirely - reserved for
Stage 5 independent validation (no data leakage).
"""
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_v1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

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


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # Reserve Tier-2 (ANSYS validation candidates) - NOT used in training at all
    train_pool = df[~df["ansys_validation_candidate"]].copy()

    X = pd.get_dummies(train_pool[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS],
                        columns=CATEGORICAL_COLS)
    y = train_pool[TARGET_COLS]
    return X, y, list(X.columns)


def train_all_targets():
    X, y, feature_names = load_and_prepare()
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    results = []
    trained_models = {}

    for target in TARGET_COLS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[target], test_size=0.15, random_state=42
        )

        model = XGBRegressor(
            n_estimators=250, max_depth=4, learning_rate=0.08,
            subsample=0.85, min_child_weight=2, random_state=42,
        )

        # 5-fold CV on training pool for a robust performance estimate
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(model, X_train, y_train, cv=kf,
                                   scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

        # Fit final model on full training split, evaluate on held-out test split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, 1, y_test.values))) * 100)

        results.append({
            "target": target,
            "cv_mae_mean": round(cv_mae.mean(), 3), "cv_mae_std": round(cv_mae.std(), 3),
            "cv_r2_mean": round(cv_r2.mean(), 3),
            "test_mae": round(mae, 3), "test_rmse": round(rmse, 3),
            "test_r2": round(r2, 3), "test_mape_pct": round(mape, 2),
        })

        trained_models[target] = model
        joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{target}.pkl"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "surrogate_eval_metrics.csv"), index=False)
    return results_df, trained_models, feature_names


if __name__ == "__main__":
    results_df, models, feature_names = train_all_targets()
    print("\n=== Stage 3: Surrogate Model Evaluation ===\n")
    print(results_df.to_string(index=False))
    print(f"\nSaved {len(models)} models + metrics to {MODEL_DIR}/")
