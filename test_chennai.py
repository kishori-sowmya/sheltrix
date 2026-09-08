import sys
from src.physics.rc_model import simulate_shelter
from src.data_modules.climate_manager import ClimateManager

cfg = {
    'length': 5.0, 'width': 4.0, 'wall_height': 2.8, 'roof_angle': 20.0, 
    'orientation_deg': 0.0, 'wall_material': 'cseb', 'roof_material': 'sandwich_panel', 
    'floor_material': 'concrete', 'insulation_material': 'xps_foam', 
    'insulation_thickness_mm': 80.0, 'window_ratio': 0.15, 'pcm_present': 1, 
    'pcm_thickness_mm': 20.0, 'glazing_type': 'double_pane'
}

cm = ClimateManager()
chennai = cm.get_profile('chennai_summer')

res = simulate_shelter(cfg, climate_dict=chennai)
print(f"Cooling Energy: {res['cooling_energy_required_kWh']} kWh")
print(f"Comfort Hours: {res['comfort_hours']}")
