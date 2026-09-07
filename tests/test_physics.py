"""
Module 17: Automated Tests for Physics Model & Heat Balance
"""

import pytest
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.rc_model import simulate_shelter, compute_layer_resistance
from src.data_modules.climate_manager import ClimateManager
from src.data_modules.material_db import MaterialDatabase


def test_layer_resistance_calculation():
    r_val = compute_layer_resistance("wall", "cseb", "xps_foam", 100.0)
    assert r_val > 0.5, "R-value must be positive and physically realistic"


def test_rc_physics_simulation_bounds():
    test_params = {
        "length": 5.0, "width": 4.0, "wall_height": 2.8, "roof_angle": 20.0, "orientation_deg": 0.0,
        "wall_material": "cseb", "roof_material": "sandwich_panel", "floor_material": "concrete",
        "insulation_material": "xps_foam", "insulation_thickness_mm": 80.0, "window_ratio": 0.15,
        "pcm_present": 1, "pcm_thickness_mm": 20.0, "glazing_type": "double_pane",
    }
    res = simulate_shelter(test_params, climate_scenario="winter")

    assert "indoor_temp_mean" in res
    assert -40.0 <= res["indoor_temp_mean"] <= 50.0, "Mean indoor temp out of physical bounds"
    assert res["comfort_hours"] >= 0 and res["comfort_hours"] <= 24, "Comfort hours must be 0-24"
    assert res["total_heat_loss_kWh"] >= 0.0, "Heat loss must be non-negative"
    assert res["heating_energy_required_kWh"] >= 0.0, "Heating energy required must be non-negative"
    assert len(res["hourly_indoor_temp"]) == 24, "Must output 24 hourly values"


def test_pcm_buffering_effect():
    params_no_pcm = {
        "length": 5.0, "width": 4.0, "wall_height": 2.8, "roof_angle": 20.0, "orientation_deg": 0.0,
        "wall_material": "cseb", "roof_material": "sandwich_panel", "floor_material": "concrete",
        "insulation_material": "xps_foam", "insulation_thickness_mm": 50.0, "window_ratio": 0.15,
        "pcm_present": 0, "pcm_thickness_mm": 0.0, "glazing_type": "single_pane",
    }
    params_with_pcm = {**params_no_pcm, "pcm_present": 1, "pcm_thickness_mm": 30.0}

    res1 = simulate_shelter(params_no_pcm, climate_scenario="shoulder")
    res2 = simulate_shelter(params_with_pcm, climate_scenario="shoulder")

    # PCM inclusion should dampen diurnal temperature swing or improve comfort
    assert res2["indoor_temp_swing"] <= res1["indoor_temp_swing"] + 0.1, "PCM must reduce/dampen temperature swing"
