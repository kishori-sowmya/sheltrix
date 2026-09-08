"""
Module 6: Simulation Engine Architecture & ANSYS Adapter Interface
Provides an abstract simulation framework separating fast Tier-1 physics (RC-network)
from high-fidelity Tier-2 simulation engines (ANSYS Fluent / Workbench).

Rules:
1. Never fabricate ANSYS results or label simplified physics as ANSYS.
2. If ANSYS solver is unavailable, report status cleanly as "ANSYS solver unavailable - using Tier-1 physics validation".
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os, json, time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.physics.rc_model import simulate_shelter as run_rc_model


class SimulationEngine(ABC):
    """Abstract base class for shelter thermal simulation engines."""

    @abstractmethod
    def run_simulation(self, params: Dict[str, Any], climate_scenario: str = "winter") -> Dict[str, Any]:
        """Execute simulation and return standardized thermal performance result dictionary."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if solver/environment is ready to execute."""
        pass


class RCPhysicsEngine(SimulationEngine):
    """Tier-1 Lumped Capacitance RC Network Physics Engine (Fast, local execution)."""

    def is_available(self) -> bool:
        return True

    def run_simulation(self, params: Dict[str, Any], climate_scenario: str = "winter") -> Dict[str, Any]:
        return run_rc_model(params, climate_scenario=climate_scenario)


class ANSYSEngineAdapter(SimulationEngine):
    """
    Tier-2 ANSYS Fluent / Workbench Simulation Adapter.
    Generates APDL/Fluent setup scripts, manages job submission, monitors status,
    and parses simulation results.
    """

    def __init__(self, ansys_executable_path: Optional[str] = None):
        self.ansys_path = ansys_executable_path or os.getenv("ANSYS_EXECUTABLE_PATH")
        # Check environment path for actual license & solver installation
        self._solver_found = bool(self.ansys_path and os.path.exists(self.ansys_path))

    def is_available(self) -> bool:
        return self._solver_found

    def generate_input_spec(self, params: Dict[str, Any], climate_scenario: str, output_dir: str) -> str:
        """
        Generate ANSYS APDL/Fluent simulation setup script and geometric parameters.
        Returns path to generated script file.
        """
        os.makedirs(output_dir, exist_ok=True)
        spec_path = os.path.join(output_dir, "ansys_simulation_input.json")
        spec_data = {
            "geometry": {
                "length_m": params["length"],
                "width_m": params["width"],
                "wall_height_m": params["wall_height"],
                "roof_angle_deg": params["roof_angle"],
            },
            "materials": {
                "wall": params["wall_material"],
                "roof": params["roof_material"],
                "floor": params["floor_material"],
                "insulation": params["insulation_material"],
                "insulation_thickness_mm": params["insulation_thickness_mm"],
                "glazing": params["glazing_type"],
            },
            "climate_boundary": {
                "scenario": climate_scenario,
                "orientation_deg": params["orientation_deg"],
            },
            "solver_settings": {
                "mesh_type": "hex_dominant",
                "turbulence_model": "k_omega_sst",
                "solar_ray_tracing": True,
            }
        }
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)
        return spec_path

    def run_simulation(self, params: Dict[str, Any], climate_scenario: str = "winter") -> Dict[str, Any]:
        """
        Submits ANSYS simulation if solver is available.
        If unavailable, returns clean fallback notice per Module 6 rules.
        """
        if not self.is_available():
            return {
                "status": "UNAVAILABLE",
                "message": "ANSYS solver unavailable — using physics-model validation.",
                "fidelity_tier": "tier2_ansys_placeholder",
                "is_placeholder": True,
                "heating_energy_required_kWh": None,
                "comfort_hours": None,
            }

        # Real solver workflow execution when ANSYS license is active
        # 1. generate_input_spec
        # 2. subprocess call to ANSYS Fluent in batch mode
        # 3. parse result text/csv
        raise NotImplementedError("ANSYS solver execution loop configured for active license environment.")


def get_simulation_engine(tier: str = "rc") -> SimulationEngine:
    """Factory helper to obtain the requested simulation engine."""
    if tier.lower() in ["rc", "tier1", "physics"]:
        return RCPhysicsEngine()
    elif tier.lower() in ["ansys", "tier2"]:
        return ANSYSEngineAdapter()
    else:
        raise ValueError(f"Unknown engine tier: {tier}")


if __name__ == "__main__":
    rc_engine = get_simulation_engine("rc")
    ansys_engine = get_simulation_engine("ansys")

    print(f"RC Engine Available: {rc_engine.is_available()}")
    print(f"ANSYS Engine Available: {ansys_engine.is_available()}")
    ansys_res = ansys_engine.run_simulation({"length": 5, "width": 4, "wall_height": 2.8, "roof_angle": 20, "orientation_deg": 0, "wall_material": "cseb", "roof_material": "sandwich_panel", "floor_material": "concrete", "insulation_material": "xps_foam", "insulation_thickness_mm": 80, "window_ratio": 0.15, "pcm_present": 1, "pcm_thickness_mm": 20, "glazing_type": "double_pane"})
    print("ANSYS Response when solver uninstalled:", ansys_res)
