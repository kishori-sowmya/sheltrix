"""
Module 4: Physics-Based Lumped Capacitance (RC-Network) Thermal Model
Simulates single-zone shelter indoor heat balance over 24 hours with a 3-day diurnal warm-up spinup.

Heat Balance Equation:
  C_total * dT/dt = Q_solar_glazing + Q_solar_wall + Q_cond_envelope + Q_cond_floor + Q_vent +/- Q_PCM + Q_heating
"""

import numpy as np
from typing import Dict, Any, List
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_modules.material_db import MATERIAL_DATABASE
from src.data_modules.climate_manager import PREDEFINED_CLIMATES

# Standard physical constants
COMFORT_LOW, COMFORT_HIGH = 18.0, 24.0   # °C ASHRAE 55 indoor thermal comfort range
AIR_CP = 1005.0                          # J/kg.K specific heat of air
AIR_RHO = 1.2                            # kg/m3 standard air density
SURFACE_FILM_R = 0.13                    # m2.K/W interior + exterior surface film resistance
TIMESTEP_S = 3600.0                      # 1 hour simulation timestep


def compute_layer_resistance(material_category: str, material_key: str, insulation_key: str, insulation_thickness_mm: float) -> float:
    """Compute total thermal resistance (R-value, m2.K/W) for an envelope wall/roof layer."""
    mat_props = MATERIAL_DATABASE[material_category][material_key]
    ins_props = MATERIAL_DATABASE["insulation"][insulation_key]

    base_wall_thickness_m = 0.20  # fixed base structural thickness
    ins_thickness_m = insulation_thickness_mm / 1000.0

    R_base = base_wall_thickness_m / mat_props["k"]
    R_ins = (ins_thickness_m / ins_props["k"]) if ins_thickness_m > 0 else 0.0
    return R_base + R_ins + SURFACE_FILM_R


def simulate_shelter(params: Dict[str, Any], climate_dict: Dict[str, Any] = None, climate_scenario: str = "winter") -> Dict[str, Any]:
    """
    Run 24-hour RC-network physics simulation for one shelter configuration.
    Includes a 3-day diurnal spinup sequence to ensure numerical equilibrium.
    """
    if climate_dict is None:
        climate_dict = PREDEFINED_CLIMATES[climate_scenario]

    T_amb_hourly = np.array(climate_dict["T_amb_hourly"])
    solar_hourly = np.array(climate_dict["solar_hourly"])

    L, W, H = params["length"], params["width"], params["wall_height"]
    floor_area = L * W
    wall_area = 2 * (L + W) * H
    roof_area = (L * W) / np.cos(np.radians(min(params["roof_angle"], 60.0)))

    window_ratio = max(0.01, min(0.60, params["window_ratio"]))
    window_area = wall_area * window_ratio
    net_wall_area = wall_area - window_area

    # Envelope thermal resistance
    R_wall = compute_layer_resistance("wall", params["wall_material"], params["insulation_material"], params["insulation_thickness_mm"])
    R_roof = compute_layer_resistance("roof", params["roof_material"], params["insulation_material"], params["insulation_thickness_mm"])
    glazing_props = MATERIAL_DATABASE["glazing"][params["glazing_type"]]

    # Thermal capacitance calculations
    air_volume = floor_area * H
    C_air = AIR_RHO * AIR_CP * air_volume

    floor_props = MATERIAL_DATABASE["floor"][params["floor_material"]]
    floor_mass_depth = 0.10  # 10 cm effective participating slab depth
    C_floor = floor_props["rho"] * floor_props["cp"] * floor_area * floor_mass_depth

    wall_props = MATERIAL_DATABASE["wall"][params["wall_material"]]
    wall_mass_depth = 0.10   # 10 cm effective participating wall depth
    C_wall = wall_props["rho"] * wall_props["cp"] * net_wall_area * wall_mass_depth

    C_total = C_air + C_floor + C_wall

    # PCM latent thermal mass buffer
    pcm_active = params.get("pcm_present", 0) == 1 and params.get("pcm_thickness_mm", 0) > 0
    if pcm_active:
        pcm_props = MATERIAL_DATABASE["pcm"]["paraffin_rt21"]
        pcm_thickness_m = params["pcm_thickness_mm"] / 1000.0
        pcm_mass = pcm_props["density"] * floor_area * pcm_thickness_m
        pcm_latent_J = pcm_mass * pcm_props["latent_heat_kJ_per_kg"] * 1000.0
        pcm_melt_T = pcm_props["melting_point_C"]
    else:
        pcm_mass, pcm_latent_J, pcm_melt_T = 0.0, 0.0, 999.0

    # Solar orientation factor (South = 0° max exposure)
    orientation_deg = params.get("orientation_deg", 0.0)
    orientation_factor = max(0.3, np.cos(np.radians(orientation_deg)))

    # Ventilation & infiltration rate (Air Changes per Hour - ACH)
    ACH = 0.3 + window_ratio * 1.5
    m_dot_air = (ACH * air_volume * AIR_RHO) / 3600.0  # kg/s

    ground_T = climate_dict.get("T_ground", T_amb_hourly.mean() - 3.0)

    # Multi-day spinup (3 days = 72 hours) to reach diurnal steady-state balance
    SPINUP_DAYS = 3
    T_curr = T_amb_hourly[0] + 5.0

    hourly_history = np.zeros(24)
    heat_loss_total = 0.0
    solar_gain_total = 0.0
    ventilation_loss_total = 0.0
    heating_energy_total = 0.0
    cooling_energy_total = 0.0
    comfort_hours_count = 0

    for day in range(SPINUP_DAYS):
        is_final_day = (day == SPINUP_DAYS - 1)
        for h in range(24):
            T_amb = T_amb_hourly[h]
            I_solar = solar_hourly[h]

            # Solar gain components
            Q_solar_window = glazing_props["SHGC"] * window_area * I_solar * orientation_factor
            Q_solar_opaque = 0.05 * wall_props["alpha"] * net_wall_area * I_solar
            Q_solar_total = Q_solar_window + Q_solar_opaque

            # Conductive heat transfers
            Q_cond_wall = (net_wall_area / R_wall) * (T_amb - T_curr)
            Q_cond_roof = (roof_area / R_roof) * (T_amb - T_curr)
            Q_cond_window = glazing_props["U_value"] * window_area * (T_amb - T_curr)
            Q_cond_floor = (floor_area / 1.2) * (ground_T - T_curr)
            Q_cond_total = Q_cond_wall + Q_cond_roof + Q_cond_window + Q_cond_floor

            # Ventilation loss
            Q_vent = m_dot_air * AIR_CP * (T_amb - T_curr)

            Q_net = Q_solar_total + Q_cond_total + Q_vent

            # Effective capacitance with PCM phase-change broadening
            effective_C = C_total
            if pcm_active and abs(T_curr - pcm_melt_T) < 2.0:
                effective_C = C_total + (pcm_latent_J / 4.0)

            dT = (Q_net * TIMESTEP_S) / effective_C
            T_uncontrolled = T_curr + dT

            # Heating & Cooling HVAC setpoint energy calculations
            heating_wh = 0.0
            cooling_wh = 0.0

            if T_uncontrolled < COMFORT_LOW:
                required_dT = COMFORT_LOW - T_uncontrolled
                Q_heater = required_dT * effective_C / TIMESTEP_S
                heating_wh = Q_heater * (TIMESTEP_S / 3600.0)
                T_curr = COMFORT_LOW
            elif T_uncontrolled > COMFORT_HIGH:
                excess_dT = T_uncontrolled - COMFORT_HIGH
                Q_cooler = excess_dT * effective_C / TIMESTEP_S
                cooling_wh = Q_cooler * (TIMESTEP_S / 3600.0)
                T_curr = COMFORT_HIGH
            else:
                T_curr = T_uncontrolled

            if is_final_day:
                hourly_history[h] = T_curr
                if heating_wh == 0.0 and cooling_wh == 0.0:
                    comfort_hours_count += 1
                
                heat_loss_total += max(0.0, -Q_cond_total) * (TIMESTEP_S / 3600.0)
                solar_gain_total += max(0.0, Q_solar_total) * (TIMESTEP_S / 3600.0)
                ventilation_loss_total += max(0.0, -Q_vent) * (TIMESTEP_S / 3600.0)
                heating_energy_total += heating_wh
                cooling_energy_total += cooling_wh

    total_hvac_energy_kWh = (heating_energy_total + cooling_energy_total) / 1000.0

    return {
        "indoor_temp_mean": float(np.mean(hourly_history)),
        "indoor_temp_min": float(np.min(hourly_history)),
        "indoor_temp_max": float(np.max(hourly_history)),
        "indoor_temp_swing": float(np.max(hourly_history) - np.min(hourly_history)),
        "comfort_hours": comfort_hours_count,
        "total_heat_loss_kWh": heat_loss_total / 1000.0,
        "total_solar_gain_kWh": solar_gain_total / 1000.0,
        "ventilation_loss_kWh": ventilation_loss_total / 1000.0,
        "heating_energy_required_kWh": heating_energy_total / 1000.0,
        "cooling_energy_required_kWh": cooling_energy_total / 1000.0,
        "total_hvac_energy_kWh": total_hvac_energy_kWh,
        "hourly_indoor_temp": hourly_history.tolist(),
        "fidelity_tier": "tier1_rc_physics",
        "assumptions": [
            "Lumped 1-zone capacitance model",
            "Diurnal 3-day warm-up spinup included",
            "Simplified PCM enthalpy buffering (+/- 2°C band around melt point)",
            "Fixed floor ground resistance R = 1.2 m2.K/W",
        ]
    }

if __name__ == "__main__":
    test_cfg = {
        "length": 5.0, "width": 4.0, "wall_height": 2.8, "roof_angle": 20.0, "orientation_deg": 0.0,
        "wall_material": "cseb", "roof_material": "sandwich_panel", "floor_material": "concrete",
        "insulation_material": "xps_foam", "insulation_thickness_mm": 80.0, "window_ratio": 0.15,
        "pcm_present": 1, "pcm_thickness_mm": 20.0, "glazing_type": "double_pane",
    }
    res = simulate_shelter(test_cfg)
    print("RC Physics Result:")
    for k, v in res.items():
        if k != "hourly_indoor_temp" and k != "assumptions":
            print(f"  {k}: {v}")
