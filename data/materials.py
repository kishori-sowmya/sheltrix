"""
Material thermal property database.
Values are standard reference values (ASHRAE Handbook of Fundamentals /
common building-physics textbooks) - order-of-magnitude realistic for
demo purposes. For a real submission cite exact datasheet sources.
"""

MATERIALS = {
    # k: thermal conductivity (W/m.K), rho: density (kg/m3), cp: specific heat (J/kg.K)
    # alpha: solar absorptivity (0-1), epsilon: emissivity (0-1)
    "brick": {"k": 0.72, "rho": 1700, "cp": 840, "alpha": 0.60, "epsilon": 0.90},
    "concrete": {"k": 1.40, "rho": 2300, "cp": 880, "alpha": 0.65, "epsilon": 0.90},
    "cseb": {"k": 0.55, "rho": 1900, "cp": 850, "alpha": 0.55, "epsilon": 0.88},  # compressed stabilized earth block
    "timber": {"k": 0.14, "rho": 550, "cp": 1600, "alpha": 0.50, "epsilon": 0.85},
    "sandwich_panel": {"k": 0.04, "rho": 45, "cp": 1000, "alpha": 0.45, "epsilon": 0.80},
}

INSULATION_MATERIALS = {
    "eps_foam": {"k": 0.034, "rho": 20, "cp": 1450},
    "xps_foam": {"k": 0.030, "rho": 35, "cp": 1450},
    "mineral_wool": {"k": 0.040, "rho": 100, "cp": 840},
    "rock_wool": {"k": 0.038, "rho": 120, "cp": 840},
}

GLAZING = {
    "single_pane": {"U_value": 5.7, "SHGC": 0.80},   # W/m2.K, solar heat gain coeff
    "double_pane": {"U_value": 2.8, "SHGC": 0.65},
}

PCM = {
    "paraffin_rt21": {
        "melting_point_C": 21,
        "latent_heat_kJ_per_kg": 190,
        "density": 880,
        "cp_solid": 2000,
        "cp_liquid": 2200,
    }
}

WALL_MATERIAL_LIST = list(MATERIALS.keys())
INSULATION_LIST = list(INSULATION_MATERIALS.keys())
GLAZING_LIST = list(GLAZING.keys())
