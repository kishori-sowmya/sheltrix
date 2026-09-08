"""
Representative hourly climate profiles for Ladakh.
Values are realistic approximations based on published climatology for
Leh/Ladakh (high-altitude cold desert): NASA POWER-style seasonal patterns.
NOTE: For a real submission, replace these with actual NASA POWER API
pulls for the target coordinates (Leh: 34.15N, 77.58E). These synthetic
but physically-reasonable profiles let development proceed without a
network dependency, and are clearly labelled as such in the dataset.
"""
import numpy as np

HOURS = np.arange(24)

def _diurnal_temp(t_mean, t_amp, phase_shift=14):
    """Simple sinusoidal diurnal temperature model, peak at ~2-3pm."""
    return t_mean + t_amp * np.cos((HOURS - phase_shift) * (2 * np.pi / 24))

def _solar_profile(peak_irradiance, sunrise=6, sunset=19):
    """Simple daylight-bounded sinusoidal solar radiation profile."""
    irr = np.zeros(24)
    day_len = sunset - sunrise
    for h in HOURS:
        if sunrise <= h <= sunset:
            frac = (h - sunrise) / day_len
            irr[h] = peak_irradiance * np.sin(np.pi * frac)
    return np.maximum(irr, 0)

try:
    from src.data_modules.climate_manager import PREDEFINED_CLIMATES
    CLIMATE_SCENARIOS = {
        k: {
            "T_amb_hourly": np.array(v["T_amb_hourly"]),
            "solar_hourly": np.array(v["solar_hourly"]),
            "wind_speed_mean": v["wind_speed_mean"],
            "T_ground": v["T_ground"],
            "RH_mean": v["RH_mean"],
        }
        for k, v in PREDEFINED_CLIMATES.items()
    }
except ImportError:
    CLIMATE_SCENARIOS = {
        "winter": {
            "T_amb_hourly": _diurnal_temp(t_mean=-10.0, t_amp=8.0),
            "solar_hourly": _solar_profile(peak_irradiance=550, sunrise=7, sunset=17),
            "wind_speed_mean": 3.5,
            "T_ground": -5.0,
            "RH_mean": 35,
        },
        "summer": {
            "T_amb_hourly": _diurnal_temp(t_mean=18.0, t_amp=10.0),
            "solar_hourly": _solar_profile(peak_irradiance=850, sunrise=5, sunset=19),
            "wind_speed_mean": 2.5,
            "T_ground": 15.0,
            "RH_mean": 25,
        },
        "shoulder": {
            "T_amb_hourly": _diurnal_temp(t_mean=4.0, t_amp=9.0),
            "solar_hourly": _solar_profile(peak_irradiance=700, sunrise=6, sunset=18),
            "wind_speed_mean": 3.0,
            "T_ground": 3.0,
            "RH_mean": 30,
        },
    }

def get_climate_summary(scenario_name):
    """Return daily summary stats used as ML features."""
    s = CLIMATE_SCENARIOS[scenario_name]
    return {
        "T_amb_mean": float(np.mean(s["T_amb_hourly"])),
        "T_amb_min": float(np.min(s["T_amb_hourly"])),
        "T_amb_max": float(np.max(s["T_amb_hourly"])),
        "solar_radiation_mean": float(np.mean(s["solar_hourly"])),
        "wind_speed_mean": s["wind_speed_mean"],
    }

if __name__ == "__main__":
    for name in CLIMATE_SCENARIOS:
        print(name, get_climate_summary(name))
