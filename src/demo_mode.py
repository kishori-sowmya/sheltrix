"""
Module 18: Fallback & Demo Mode Controller
Guarantees 100% offline functionality for SIH presentation.
Ensures zero dependency on internet connection, external APIs, or active ANSYS licenses.
"""

import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DemoModeConfig:
    DEMO_MODE_ENABLED = True
    PRIMARY_REGION = "Ladakh"
    DEFAULT_SCENARIO = "winter"
    OFFLINE_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "raw", "dataset_v1.csv")
    OFFLINE_MODEL_DIR = os.path.join(BASE_DIR, "models", "v1.0")

    @classmethod
    def get_status(cls):
        return {
            "demo_mode_active": cls.DEMO_MODE_ENABLED,
            "region": cls.PRIMARY_REGION,
            "offline_dataset_exists": os.path.exists(cls.OFFLINE_DATASET_PATH),
            "offline_models_exist": os.path.exists(os.path.join(cls.OFFLINE_MODEL_DIR, "xgb_heating_energy_required_kWh.pkl")),
            "ansys_mode": "Offline Adapter (Fallback Active)",
        }


if __name__ == "__main__":
    print("Demo Mode Status:", DemoModeConfig.get_status())
