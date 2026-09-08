"""
Simplified lumped-capacitance (RC-network) thermal model for a single-zone
shelter. This is the Tier-1 fast physics engine described in Stage 1/2:
  C_indoor * dT/dt = Q_solar + Q_cond_in - Q_cond_out - Q_vent +/- Q_PCM

Solved via explicit time-stepping over 24 hours using hourly climate
forcing. This stands in for ANSYS during bulk dataset generation; a small
stratified subset of these configurations is meant to be separately run
in ANSYS for Stage 5 validation.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.materials import MATERIALS, INSULATION_MATERIALS, GLAZING, PCM
from data.climate_ladakh import CLIMATE_SCENARIOS

COMFORT_LOW, COMFORT_HIGH = 18.0, 24.0   # deg C comfort band (ASHRAE-style fixed band for MVP)
AIR_CP = 1005      # J/kg.K
AIR_RHO = 1.2      # kg/m3 (approx at altitude-adjusted density could be refined)
SURFACE_FILM_R = 0.13  # m2.K/W combined internal+external film resistance (approx)
TIMESTEP_S = 3600  # 1 hour


def compute_envelope_resistance(wall_material, insulation_thickness_mm, insulation_material):
    """R-value of wall assembly: base wall material (fixed 0.2m thickness) + insulation layer."""
    wall_props = MATERIALS[wall_material]
    ins_props = INSULATION_MATERIALS[insulation_material]
    wall_thickness = 0.20  # m, fixed base wall thickness assumption for MVP
    ins_thickness = insulation_thickness_mm / 1000.0

    R_wall = wall_thickness / wall_props["k"]
    R_ins = ins_thickness / ins_props["k"] if ins_thickness > 0 else 0
    R_total = R_wall + R_ins + SURFACE_FILM_R
    return R_total


def simulate_shelter(params, climate_scenario="winter"):
    """
    Run a 24-hour RC-network simulation for one shelter configuration.

    params: dict with keys -
        length, width, wall_height, roof_angle, orientation_deg,
        wall_material, roof_material, floor_material, insulation_material,
        insulation_thickness_mm, window_ratio, pcm_present, pcm_thickness_mm,
        glazing_type, ach (air changes per hour, derived from window_ratio)

    Returns dict of daily performance outputs.
    """
    climate = CLIMATE_SCENARIOS[climate_scenario]
    T_amb_hourly = climate["T_amb_hourly"]
    solar_hourly = climate["solar_hourly"]

    L, W, H = params["length"], params["width"], params["wall_height"]
    floor_area = L * W
    wall_area = 2 * (L + W) * H
    roof_area = (L * W) / np.cos(np.radians(min(params["roof_angle"], 60)))  # sloped roof area correction

    window_area = wall_area * params["window_ratio"]
    net_wall_area = wall_area - window_area

    # Envelope thermal resistance
    R_wall = compute_envelope_resistance(params["wall_material"], params["insulation_thickness_mm"],
                                          params["insulation_material"])
    R_roof = compute_envelope_resistance(params["roof_material"], params["insulation_thickness_mm"],
                                          params["insulation_material"])
    glazing = GLAZING[params["glazing_type"]]

    # Indoor air + thermal mass capacitance
    air_volume = floor_area * H
    C_air = AIR_RHO * AIR_CP * air_volume
    floor_props = MATERIALS[params["floor_material"]]
    floor_thickness_thermal_mass = 0.10  # m, effective participating floor slab depth
    C_floor = floor_props["rho"] * floor_props["cp"] * floor_area * floor_thickness_thermal_mass
    C_total = C_air + C_floor

    # PCM effective extra capacitance (simplified: adds heat capacity near melting point)
    pcm_active = params.get("pcm_present", 0) == 1
    if pcm_active:
        pcm_props = PCM["paraffin_rt21"]
        pcm_thickness_m = params["pcm_thickness_mm"] / 1000.0
        pcm_mass = pcm_props["density"] * floor_area * pcm_thickness_m
        pcm_latent_J = pcm_mass * pcm_props["latent_heat_kJ_per_kg"] * 1000
        pcm_melt_T = pcm_props["melting_point_C"]
    else:
        pcm_mass, pcm_latent_J, pcm_melt_T = 0, 0, 999

    # Orientation factor: simple cosine-based solar exposure multiplier for the
    # window-bearing facade (assume windows face 'orientation_deg' from south=0)
    orientation_factor = max(0.3, np.cos(np.radians(params["orientation_deg"])))

    # Air changes per hour (infiltration/ventilation), scaled with window ratio
    ACH = 0.3 + params["window_ratio"] * 1.5  # heuristic: more openings -> more infiltration
    m_dot_air = (ACH * air_volume * AIR_RHO) / 3600.0  # kg/s

    ground_T = climate["T_ground"]

    # --- Time stepping (explicit Euler, 1-hour steps) ---
    T_indoor = np.zeros(24)
    T_indoor[0] = T_amb_hourly[0] + 5  # initial guess, slightly warmer than outside
    pcm_energy_stored = 0.0

    heat_loss_total = 0.0
    solar_gain_total = 0.0
    heating_energy_total = 0.0

    for h in range(24):
        T_amb = T_amb_hourly[h]
        I_solar = solar_hourly[h]
        T_prev = T_indoor[h - 1] if h > 0 else T_indoor[0]

        # Solar gain through windows
        Q_solar = glazing["SHGC"] * window_area * I_solar * orientation_factor
        # Solar gain absorbed on opaque wall/roof surface then partially conducted in (simplified as small added term)
        Q_solar_wall = 0.05 * MATERIALS[params["wall_material"]]["alpha"] * net_wall_area * I_solar

        # Conductive losses/gains through walls, roof, and windows (U-value based)
        Q_cond_wall = (net_wall_area / R_wall) * (T_amb - T_prev)
        Q_cond_roof = (roof_area / R_roof) * (T_amb - T_prev)
        Q_cond_window = glazing["U_value"] * window_area * (T_amb - T_prev)
        Q_cond_floor = (floor_area / 1.2) * (ground_T - T_prev)  # approx floor R=1.2 m2K/W

        Q_cond_total = Q_cond_wall + Q_cond_roof + Q_cond_window + Q_cond_floor

        # Ventilation loss
        Q_vent = m_dot_air * AIR_CP * (T_amb - T_prev)

        # Net heat into zone (W), before heating input
        Q_net = Q_solar + Q_solar_wall + Q_cond_total + Q_vent

        # PCM buffering: if crossing melting point, absorb/release energy instead of raising temp
        effective_C = C_total
        if pcm_active and abs(T_prev - pcm_melt_T) < 2.0:
            effective_C = C_total + (pcm_latent_J / 4.0)  # smeared latent effect over ~4C band

        dT = (Q_net * TIMESTEP_S) / effective_C
        T_new = T_prev + dT

        # Simple heating system: if below comfort floor, add heat to bring to COMFORT_LOW (a "smart" heater)
        heating_energy_wh = 0.0
        if T_new < COMFORT_LOW:
            required_dT = COMFORT_LOW - T_new
            Q_heater = required_dT * effective_C / TIMESTEP_S
            heating_energy_wh = Q_heater * (TIMESTEP_S / 3600.0)
            T_new = COMFORT_LOW

        T_indoor[h] = T_new
        heat_loss_total += max(0, -Q_cond_total) * (TIMESTEP_S / 3600.0)
        solar_gain_total += (Q_solar + Q_solar_wall) * (TIMESTEP_S / 3600.0)
        heating_energy_total += heating_energy_wh

    comfort_hours = int(np.sum((T_indoor >= COMFORT_LOW) & (T_indoor <= COMFORT_HIGH + 4)))
    # (+4 upper slack: winter demo rarely overheats; comfort mainly limited by cold)

    return {
        "indoor_temp_mean": float(np.mean(T_indoor)),
        "indoor_temp_min": float(np.min(T_indoor)),
        "indoor_temp_max": float(np.max(T_indoor)),
        "indoor_temp_swing": float(np.max(T_indoor) - np.min(T_indoor)),
        "comfort_hours": comfort_hours,
        "total_heat_loss_kWh": heat_loss_total / 1000.0,
        "total_solar_gain_kWh": solar_gain_total / 1000.0,
        "heating_energy_required_kWh": heating_energy_total / 1000.0,
    }


if __name__ == "__main__":
    test_params = {
        "length": 5, "width": 4, "wall_height": 2.8, "roof_angle": 20, "orientation_deg": 0,
        "wall_material": "cseb", "roof_material": "sandwich_panel", "floor_material": "concrete",
        "insulation_material": "xps_foam", "insulation_thickness_mm": 80, "window_ratio": 0.15,
        "pcm_present": 1, "pcm_thickness_mm": 20, "glazing_type": "double_pane",
    }
    result = simulate_shelter(test_params, climate_scenario="winter")
    for k, v in result.items():
        print(f"{k}: {v:.2f}")
