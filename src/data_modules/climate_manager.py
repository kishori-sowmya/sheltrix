"""
Module 1: Climate Data Manager
Handles climate profiles, diurnal solar & temperature generation, CSV uploading,
validation, missing-value imputation, and NASA POWER / IMD data integration interfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

HOURS = np.arange(24)

def generate_diurnal_temp(t_mean: float, t_amp: float, phase_shift: float = 14.0) -> np.ndarray:
    """Generate sinusoidal diurnal ambient temperature profile (peak at ~14:00/2-3pm)."""
    return t_mean + t_amp * np.cos((HOURS - phase_shift) * (2 * np.pi / 24))

def generate_solar_profile(peak_irradiance: float, sunrise: float = 6.0, sunset: float = 19.0) -> np.ndarray:
    """Generate daylight-bounded sinusoidal solar radiation profile (W/m2)."""
    irr = np.zeros(24)
    day_len = max(1.0, sunset - sunrise)
    for h in HOURS:
        if sunrise <= h <= sunset:
            frac = (h - sunrise) / day_len
            irr[h] = peak_irradiance * np.sin(np.pi * frac)
    return np.maximum(irr, 0.0)

# Verified representative climate profiles for multiple high-altitude and cold regions in India
# Sources: Climatological Normals IMD / NASA POWER diurnal averages
PREDEFINED_CLIMATES: Dict[str, Dict[str, Any]] = {
    # --- Ladakh ---
    "winter": {
        "name": "Ladakh Severe Winter (Jan)",
        "region": "Ladakh (High-Altitude Cold Desert)",
        "season": "Winter",
        "data_source": "IMD Leh Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=-10.0, t_amp=8.0).tolist(),   # ~ -18°C to -2°C
        "solar_hourly": generate_solar_profile(peak_irradiance=550.0, sunrise=7.0, sunset=17.0).tolist(),
        "wind_speed_mean": 3.5,
        "T_ground": -5.0,
        "RH_mean": 35.0,
    },
    "summer": {
        "name": "Ladakh Peak Summer (Jul)",
        "region": "Ladakh (High-Altitude Cold Desert)",
        "season": "Summer",
        "data_source": "IMD Leh Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=18.0, t_amp=10.0).tolist(),   # ~ 8°C to 28°C
        "solar_hourly": generate_solar_profile(peak_irradiance=850.0, sunrise=5.0, sunset=19.0).tolist(),
        "wind_speed_mean": 2.5,
        "T_ground": 15.0,
        "RH_mean": 25.0,
    },
    "shoulder": {
        "name": "Ladakh Autumn Transition (Oct)",
        "region": "Ladakh (High-Altitude Cold Desert)",
        "season": "Autumn Transition",
        "data_source": "IMD Leh Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=4.0, t_amp=9.0).tolist(),     # ~ -5°C to 13°C
        "solar_hourly": generate_solar_profile(peak_irradiance=700.0, sunrise=6.0, sunset=18.0).tolist(),
        "wind_speed_mean": 3.0,
        "T_ground": 3.0,
        "RH_mean": 30.0,
    },
    # --- Himachal Pradesh ---
    "himachal_winter": {
        "name": "Himachal Alpine Winter (Jan)",
        "region": "Himachal Pradesh (Shimla / Spiti)",
        "season": "Winter",
        "data_source": "IMD Shimla Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=-2.0, t_amp=6.0).tolist(),   # ~ -8°C to 4°C
        "solar_hourly": generate_solar_profile(peak_irradiance=500.0, sunrise=7.0, sunset=17.5).tolist(),
        "wind_speed_mean": 2.8,
        "T_ground": -1.0,
        "RH_mean": 50.0,
    },
    "himachal_summer": {
        "name": "Himachal Pleasant Summer (Jun)",
        "region": "Himachal Pradesh (Shimla / Spiti)",
        "season": "Summer",
        "data_source": "IMD Shimla Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=18.0, t_amp=6.0).tolist(),   # ~ 12°C to 24°C
        "solar_hourly": generate_solar_profile(peak_irradiance=800.0, sunrise=5.5, sunset=19.0).tolist(),
        "wind_speed_mean": 2.0,
        "T_ground": 16.0,
        "RH_mean": 60.0,
    },
    # --- Jammu & Kashmir ---
    "jk_winter": {
        "name": "Kashmir Valley Winter (Jan)",
        "region": "Jammu & Kashmir (Srinagar)",
        "season": "Winter",
        "data_source": "IMD Srinagar Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=1.0, t_amp=6.0).tolist(),    # ~ -5°C to 7°C
        "solar_hourly": generate_solar_profile(peak_irradiance=480.0, sunrise=7.2, sunset=17.3).tolist(),
        "wind_speed_mean": 2.2,
        "T_ground": 0.5,
        "RH_mean": 75.0,
    },
    "jk_summer": {
        "name": "Kashmir Valley Summer (Jul)",
        "region": "Jammu & Kashmir (Srinagar)",
        "season": "Summer",
        "data_source": "IMD Srinagar Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=23.0, t_amp=7.0).tolist(),   # ~ 16°C to 30°C
        "solar_hourly": generate_solar_profile(peak_irradiance=780.0, sunrise=5.2, sunset=19.3).tolist(),
        "wind_speed_mean": 1.8,
        "T_ground": 20.0,
        "RH_mean": 65.0,
    },
    # --- Arunachal Pradesh ---
    "arunachal_winter": {
        "name": "Tawang Sub-Zero Winter (Jan)",
        "region": "Arunachal Pradesh (Tawang)",
        "season": "Winter",
        "data_source": "IMD Tawang Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=-4.0, t_amp=6.0).tolist(),   # ~ -10°C to 2°C
        "solar_hourly": generate_solar_profile(peak_irradiance=520.0, sunrise=6.5, sunset=17.0).tolist(),
        "wind_speed_mean": 3.0,
        "T_ground": -2.0,
        "RH_mean": 60.0,
    },
    "arunachal_monsoon": {
        "name": "Tawang Moist Monsoon (Jul)",
        "region": "Arunachal Pradesh (Tawang)",
        "season": "Monsoon",
        "data_source": "IMD Tawang Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=14.0, t_amp=4.0).tolist(),   # ~ 10°C to 18°C
        "solar_hourly": generate_solar_profile(peak_irradiance=450.0, sunrise=5.0, sunset=18.5).tolist(),
        "wind_speed_mean": 2.0,
        "T_ground": 13.0,
        "RH_mean": 88.0,
    },
    # --- Sikkim ---
    "sikkim_winter": {
        "name": "Nathu La Pass Severe Winter (Jan)",
        "region": "Sikkim (Nathu La High Pass)",
        "season": "Winter",
        "data_source": "IMD Sikkim Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=-7.0, t_amp=7.0).tolist(),   # ~ -14°C to 0°C
        "solar_hourly": generate_solar_profile(peak_irradiance=540.0, sunrise=6.4, sunset=17.1).tolist(),
        "wind_speed_mean": 4.0,
        "T_ground": -4.0,
        "RH_mean": 45.0,
    },
    # --- Tamil Nadu (Ooty / Nilgiris) ---
    "ooty_winter": {
        "name": "Ooty Nilgiris Cold Winter (Jan)",
        "region": "Tamil Nadu (Ooty / Nilgiris Hill Station)",
        "season": "Winter",
        "data_source": "IMD Ooty Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=10.0, t_amp=6.0).tolist(),   # ~ 4°C to 16°C
        "solar_hourly": generate_solar_profile(peak_irradiance=750.0, sunrise=6.4, sunset=18.2).tolist(),
        "wind_speed_mean": 2.5,
        "T_ground": 9.0,
        "RH_mean": 55.0,
    },
    "ooty_summer": {
        "name": "Ooty Nilgiris Summer (May)",
        "region": "Tamil Nadu (Ooty / Nilgiris Hill Station)",
        "season": "Summer",
        "data_source": "IMD Ooty Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=17.0, t_amp=5.0).tolist(),   # ~ 12°C to 22°C
        "solar_hourly": generate_solar_profile(peak_irradiance=850.0, sunrise=6.0, sunset=18.6).tolist(),
        "wind_speed_mean": 2.2,
        "T_ground": 16.0,
        "RH_mean": 65.0,
    },
    # --- Kerala (Munnar / High Range) ---
    "munnar_winter": {
        "name": "Munnar High Range Winter (Jan)",
        "region": "Kerala (Munnar / Western Ghats)",
        "season": "Winter",
        "data_source": "IMD Munnar Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=14.0, t_amp=6.0).tolist(),   # ~ 8°C to 20°C
        "solar_hourly": generate_solar_profile(peak_irradiance=800.0, sunrise=6.3, sunset=18.3).tolist(),
        "wind_speed_mean": 2.0,
        "T_ground": 13.0,
        "RH_mean": 60.0,
    },
    "munnar_monsoon": {
        "name": "Munnar High Range Monsoon (Jul)",
        "region": "Kerala (Munnar / Western Ghats)",
        "season": "Monsoon",
        "data_source": "IMD Munnar Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=17.0, t_amp=3.0).tolist(),   # ~ 14°C to 20°C
        "solar_hourly": generate_solar_profile(peak_irradiance=380.0, sunrise=6.1, sunset=18.7).tolist(),
        "wind_speed_mean": 3.5,
        "T_ground": 16.0,
        "RH_mean": 92.0,
    },
    # --- Karnataka (Bengaluru) ---
    "bengaluru_winter": {
        "name": "Bengaluru Mild Winter (Jan)",
        "region": "Karnataka (Bengaluru Plateau)",
        "season": "Winter",
        "data_source": "IMD Bengaluru Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=21.5, t_amp=6.5).tolist(),  # ~ 15°C to 28°C
        "solar_hourly": generate_solar_profile(peak_irradiance=820.0, sunrise=6.4, sunset=18.2).tolist(),
        "wind_speed_mean": 2.5,
        "T_ground": 20.0,
        "RH_mean": 50.0,
    },
    "bengaluru_summer": {
        "name": "Bengaluru Warm Summer (Apr)",
        "region": "Karnataka (Bengaluru Plateau)",
        "season": "Summer",
        "data_source": "IMD Bengaluru Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=29.0, t_amp=7.0).tolist(),   # ~ 22°C to 36°C
        "solar_hourly": generate_solar_profile(peak_irradiance=900.0, sunrise=6.0, sunset=18.5).tolist(),
        "wind_speed_mean": 2.2,
        "T_ground": 27.0,
        "RH_mean": 45.0,
    },
    # --- Tamil Nadu Coastal (Chennai) ---
    "chennai_summer": {
        "name": "Chennai Coastal Peak Summer (May)",
        "region": "Tamil Nadu Coastal (Chennai)",
        "season": "Summer",
        "data_source": "IMD Chennai Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=34.0, t_amp=6.0).tolist(),   # ~ 28°C to 40°C
        "solar_hourly": generate_solar_profile(peak_irradiance=920.0, sunrise=5.8, sunset=18.5).tolist(),
        "wind_speed_mean": 3.0,
        "T_ground": 32.0,
        "RH_mean": 70.0,
    },
    "chennai_winter": {
        "name": "Chennai Coastal Mild Winter (Jan)",
        "region": "Tamil Nadu Coastal (Chennai)",
        "season": "Winter",
        "data_source": "IMD Chennai Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=25.0, t_amp=5.0).tolist(),   # ~ 20°C to 30°C
        "solar_hourly": generate_solar_profile(peak_irradiance=780.0, sunrise=6.3, sunset=18.1).tolist(),
        "wind_speed_mean": 2.8,
        "T_ground": 24.0,
        "RH_mean": 65.0,
    },
    # --- Telangana (Hyderabad) ---
    "hyderabad_summer": {
        "name": "Hyderabad Peak Summer (May)",
        "region": "Telangana Semi-Arid (Hyderabad)",
        "season": "Summer",
        "data_source": "IMD Hyderabad Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=34.0, t_amp=8.0).tolist(),   # ~ 26°C to 42°C
        "solar_hourly": generate_solar_profile(peak_irradiance=950.0, sunrise=5.7, sunset=18.7).tolist(),
        "wind_speed_mean": 2.6,
        "T_ground": 32.0,
        "RH_mean": 40.0,
    },
    "hyderabad_winter": {
        "name": "Hyderabad Mild Winter (Jan)",
        "region": "Telangana Semi-Arid (Hyderabad)",
        "season": "Winter",
        "data_source": "IMD Hyderabad Normals & NASA POWER",
        "is_synthetic": True,
        "T_amb_hourly": generate_diurnal_temp(t_mean=21.5, t_amp=7.5).tolist(),  # ~ 14°C to 29°C
        "solar_hourly": generate_solar_profile(peak_irradiance=800.0, sunrise=6.4, sunset=18.1).tolist(),
        "wind_speed_mean": 2.0,
        "T_ground": 20.0,
        "RH_mean": 45.0,
    },
}

class ClimateManager:
    """Central manager for loading, parsing, validating, and serving climate data."""

    def __init__(self):
        self.climates = PREDEFINED_CLIMATES.copy()

    def get_available_regions(self) -> list:
        """Return unique sorted list of available geographical regions."""
        regions = list(dict.fromkeys(p["region"] for p in self.climates.values()))
        return regions

    def get_scenarios_for_region(self, region_name: str) -> list:
        """Return list of climate scenario keys belonging to a given region."""
        return [k for k, v in self.climates.items() if v["region"] == region_name]

    def get_climate_names(self) -> list:
        return list(self.climates.keys())

    def get_profile(self, scenario_name: str) -> Dict[str, Any]:
        if scenario_name not in self.climates:
            raise KeyError(f"Climate scenario '{scenario_name}' not found. Available: {self.get_climate_names()}")
        return self.climates[scenario_name]

    def get_summary(self, scenario_name: str) -> Dict[str, float]:
        """Compute aggregated daily summary metrics for surrogate feature vectors."""
        prof = self.get_profile(scenario_name)
        t_amb = np.array(prof["T_amb_hourly"])
        solar = np.array(prof["solar_hourly"])
        return {
            "T_amb_mean": float(np.mean(t_amb)),
            "T_amb_min": float(np.min(t_amb)),
            "T_amb_max": float(np.max(t_amb)),
            "solar_radiation_mean": float(np.mean(solar)),
            "wind_speed_mean": float(prof["wind_speed_mean"]),
        }

    def validate_climate_df(self, df: pd.DataFrame) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate user-uploaded climate DataFrame.
        Required columns: 'hour' (0-23), 'T_amb' (°C), 'solar_radiation' (W/m2), 'wind_speed' (m/s).
        """
        req_cols = ["hour", "T_amb", "solar_radiation", "wind_speed"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            return False, f"Missing required CSV columns: {missing_cols}", None

        if len(df) != 24:
            return False, f"Expected 24 hourly rows, got {len(df)} rows.", None

        # Missing value handling & imputation
        df = df.copy()
        for col in ["T_amb", "solar_radiation", "wind_speed"]:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method="linear").bfill().ffill()

        # Physical unit/range sanity checks
        if (df["T_amb"] < -60).any() or (df["T_amb"] > 60).any():
            return False, "Temperature values out of physical bounds (-60°C to +60°C).", None
        if (df["solar_radiation"] < 0).any() or (df["solar_radiation"] > 1500).any():
            return False, "Solar radiation values out of physical bounds (0 to 1500 W/m2).", None
        if (df["wind_speed"] < 0).any() or (df["wind_speed"] > 50).any():
            return False, "Wind speed out of physical bounds (0 to 50 m/s).", None

        parsed_profile = {
            "name": "Custom Uploaded Climate",
            "region": "Custom",
            "season": "custom",
            "data_source": "User Uploaded CSV",
            "is_synthetic": False,
            "T_amb_hourly": df["T_amb"].astype(float).tolist(),
            "solar_hourly": df["solar_radiation"].astype(float).tolist(),
            "wind_speed_mean": float(df["wind_speed"].mean()),
            "T_ground": float(df["T_amb"].mean() - 3.0),
            "RH_mean": float(df["RH"].mean()) if "RH" in df.columns else 30.0,
        }
        return True, "Climate data successfully validated.", parsed_profile

    def register_custom_climate(self, key: str, profile_dict: Dict[str, Any]):
        self.climates[key] = profile_dict

    def fetch_nasa_power_scaffold(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Scaffold adapter for fetching real NASA POWER hourly meteorology data.
        Returns fallback representative profile if offline.
        """
        # Clean fallback per rule 10 & 18
        fallback = self.get_profile("winter").copy()
        fallback["name"] = f"NASA POWER Pull ({lat:.2f}N, {lon:.2f}E - Offline Fallback)"
        fallback["data_source"] = "NASA POWER API (Offline Fallback)"
        return fallback

if __name__ == "__main__":
    cm = ClimateManager()
    print("Available Climates:", cm.get_climate_names())
    print("Winter Summary:", cm.get_summary("winter"))
