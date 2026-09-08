"""
Module 2: Material Database
Structured database of thermal and physical properties for structural walls, roof assemblies,
floor slabs, envelope insulations, window glazings, and phase change materials (PCM).
All scientific properties cited from standard references (ASHRAE Fundamentals, ISO 10456, IS 3792).
"""

from typing import Dict, Any, List, Optional

# Enriched scientific material dataset
MATERIAL_DATABASE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "wall": {
        "brick": {
            "name": "Fired Clay Brick Masonry",
            "category": "wall",
            "k": 0.72,              # W/m.K
            "rho": 1700,            # kg/m3
            "cp": 840,              # J/kg.K
            "alpha": 0.60,          # Solar absorptivity (0-1)
            "epsilon": 0.90,        # Thermal emissivity (0-1)
            "min_thickness_mm": 100,
            "max_thickness_mm": 300,
            "cost_per_m2_base": 3800.0, # INR / m2 per 100mm
            "source": "ASHRAE Handbook - Fundamentals (2021) Ch. 26",
            "notes": "Traditional burnt clay brick; moderate thermal mass.",
            "requires_validation": False,
        },
        "concrete": {
            "name": "Dense Reinforced Concrete",
            "category": "wall",
            "k": 1.40,
            "rho": 2300,
            "cp": 880,
            "alpha": 0.65,
            "epsilon": 0.90,
            "min_thickness_mm": 100,
            "max_thickness_mm": 250,
            "cost_per_m2_base": 4600.0, # INR / m2 per 100mm
            "source": "ISO 10456:2007 Building Materials",
            "notes": "High structural strength and high thermal mass, but poor insulation.",
            "requires_validation": False,
        },
        "cseb": {
            "name": "Compressed Stabilized Earth Block (CSEB)",
            "category": "wall",
            "k": 0.55,
            "rho": 1900,
            "cp": 850,
            "alpha": 0.55,
            "epsilon": 0.88,
            "min_thickness_mm": 150,
            "max_thickness_mm": 300,
            "cost_per_m2_base": 2500.0,
            "source": "Auroville Earth Institute / IS 1725:2013",
            "notes": "Low embodied energy, local soil utilization, excellent thermal capacity.",
            "requires_validation": False,
        },
        "timber": {
            "name": "Softwood Construction Timber",
            "category": "wall",
            "k": 0.14,
            "rho": 550,
            "cp": 1600,
            "alpha": 0.50,
            "epsilon": 0.85,
            "min_thickness_mm": 50,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 5000.0,
            "source": "ASHRAE Handbook - Fundamentals (2021)",
            "notes": "Low thermal conductivity, light weight; requires fire rating treatment.",
            "requires_validation": False,
        },
        "sandwich_panel": {
            "name": "Insulated Metal Sandwich Panel (PUF Core)",
            "category": "wall",
            "k": 0.04,
            "rho": 45,
            "cp": 1000,
            "alpha": 0.45,
            "epsilon": 0.80,
            "min_thickness_mm": 50,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 3400.0,
            "source": "Manufacturer Technical Datasheet (Pre-fabricated Modular)",
            "notes": "Modular high-performance prefab wall panel for rapid assembly.",
            "requires_validation": False,
        },
    },
    "roof": {
        "brick": {
            "name": "Jack-Arch Brick Roof",
            "category": "roof",
            "k": 0.72,
            "rho": 1700,
            "cp": 840,
            "alpha": 0.60,
            "epsilon": 0.90,
            "min_thickness_mm": 100,
            "max_thickness_mm": 250,
            "cost_per_m2_base": 4200.0,
            "source": "IS 3792: Thermal Design of Buildings",
            "notes": "Traditional arched brick roof construction.",
            "requires_validation": False,
        },
        "concrete": {
            "name": "RCC Roof Slab",
            "category": "roof",
            "k": 1.40,
            "rho": 2300,
            "cp": 880,
            "alpha": 0.65,
            "epsilon": 0.90,
            "min_thickness_mm": 120,
            "max_thickness_mm": 200,
            "cost_per_m2_base": 5500.0,
            "source": "IS 456: Code of Practice for Plain and Reinforced Concrete",
            "notes": "Standard flat concrete slab.",
            "requires_validation": False,
        },
        "cseb": {
            "name": "CSEB Vaulted Roof",
            "category": "roof",
            "k": 0.55,
            "rho": 1900,
            "cp": 850,
            "alpha": 0.55,
            "epsilon": 0.88,
            "min_thickness_mm": 150,
            "max_thickness_mm": 250,
            "cost_per_m2_base": 3000.0,
            "source": "Auroville Earth Institute Guidelines",
            "notes": "Eco-friendly earth vault roof.",
            "requires_validation": False,
        },
        "timber": {
            "name": "Timber Joist & Decking Roof",
            "category": "roof",
            "k": 0.14,
            "rho": 550,
            "cp": 1600,
            "alpha": 0.50,
            "epsilon": 0.85,
            "min_thickness_mm": 75,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 5800.0,
            "source": "ASHRAE Fundamentals",
            "notes": "Traditional wooden joist roof with mud/poplar decking.",
            "requires_validation": False,
        },
        "sandwich_panel": {
            "name": "Insulated Metal Roof Sandwich Panel",
            "category": "roof",
            "k": 0.04,
            "rho": 45,
            "cp": 1000,
            "alpha": 0.45,
            "epsilon": 0.80,
            "min_thickness_mm": 50,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 3600.0,
            "source": "Industrial Building Systems Standard",
            "notes": "Lightweight roof panel with high thermal resistance.",
            "requires_validation": False,
        },
    },
    "floor": {
        "concrete": {
            "name": "Slab-on-Grade Concrete Floor",
            "category": "floor",
            "k": 1.40,
            "rho": 2300,
            "cp": 880,
            "alpha": 0.70,
            "epsilon": 0.90,
            "min_thickness_mm": 100,
            "max_thickness_mm": 200,
            "cost_per_m2_base": 3000.0,
            "source": "ASHRAE Fundamentals",
            "notes": "Direct ground contact slab providing thermal storage coupling.",
            "requires_validation": False,
        },
        "brick": {
            "name": "Brick Paver Floor",
            "category": "floor",
            "k": 0.72,
            "rho": 1700,
            "cp": 840,
            "alpha": 0.65,
            "epsilon": 0.90,
            "min_thickness_mm": 75,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 2400.0,
            "source": "IS 3792",
            "notes": "Local brick paver flooring.",
            "requires_validation": False,
        },
        "cseb": {
            "name": "Stabilized Rammed Earth Floor",
            "category": "floor",
            "k": 0.55,
            "rho": 1900,
            "cp": 850,
            "alpha": 0.60,
            "epsilon": 0.88,
            "min_thickness_mm": 100,
            "max_thickness_mm": 200,
            "cost_per_m2_base": 1700.0,
            "source": "Auroville Earth Institute",
            "notes": "Rammed earth floor slab.",
            "requires_validation": False,
        },
    },
    "insulation": {
        "eps_foam": {
            "name": "Expanded Polystyrene (EPS)",
            "category": "insulation",
            "k": 0.034,
            "rho": 20,
            "cp": 1450,
            "min_thickness_mm": 10,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 1000.0, # per 50mm
            "source": "ISO 10456",
            "notes": "Cost-effective lightweight insulation board.",
            "requires_validation": False,
        },
        "xps_foam": {
            "name": "Extruded Polystyrene (XPS)",
            "category": "insulation",
            "k": 0.030,
            "rho": 35,
            "cp": 1450,
            "min_thickness_mm": 10,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 1500.0,
            "source": "ISO 10456",
            "notes": "High moisture resistance and superior R-value.",
            "requires_validation": False,
        },
        "mineral_wool": {
            "name": "Mineral Wool Batts",
            "category": "insulation",
            "k": 0.040,
            "rho": 100,
            "cp": 840,
            "min_thickness_mm": 25,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 1250.0,
            "source": "ASHRAE Fundamentals",
            "notes": "Non-combustible fire-safe envelope insulation.",
            "requires_validation": False,
        },
        "rock_wool": {
            "name": "Rock Wool Board",
            "category": "insulation",
            "k": 0.038,
            "rho": 120,
            "cp": 840,
            "min_thickness_mm": 25,
            "max_thickness_mm": 150,
            "cost_per_m2_base": 1700.0,
            "source": "ASHRAE Fundamentals",
            "notes": "High density acoustic and thermal insulation.",
            "requires_validation": False,
        },
        },
    "glazing": {
        "single_pane": {
            "name": "Single Clear Glass Pane (6mm)",
            "category": "glazing",
            "U_value": 5.7, # W/m2.K
            "SHGC": 0.80,   # Solar Heat Gain Coefficient
            "cost_per_m2_base": 2500.0,
            "source": "NFRC 100/200 Glazing Directory",
            "notes": "Basic single glass pane, low thermal resistance.",
            "requires_validation": False,
        },
        "double_pane": {
            "name": "Double Glazed Unit (6-12-6 Air Filled)",
            "category": "glazing",
            "U_value": 2.8,
            "SHGC": 0.65,
            "cost_per_m2_base": 6200.0,
            "source": "NFRC 100/200 Glazing Directory",
            "notes": "Sealed double glazing unit for cold climate passive heating.",
            "requires_validation": False,
        },
    },
    "pcm": {
        "paraffin_rt21": {
            "name": "Rubitherm RT21 Organic Paraffin PCM",
            "category": "pcm",
            "melting_point_C": 21.0,
            "latent_heat_kJ_per_kg": 190.0,
            "density": 880,
            "cp_solid": 2000,
            "cp_liquid": 2200,
            "min_thickness_mm": 5,
            "max_thickness_mm": 50,
            "cost_per_kg": 1200.0,
            "source": "Rubitherm Technologies GmbH Technical Datasheet",
            "notes": "Phase change thermal storage buffer centered around 21°C comfort setpoint.",
            "requires_validation": False,
        }
    }
}

class MaterialDatabase:
    """Manager for querying, filtering, and validating construction material properties."""

    def __init__(self):
        self.db = MATERIAL_DATABASE

    def get_categories(self) -> List[str]:
        return list(self.db.keys())

    def get_materials_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        if category not in self.db:
            raise KeyError(f"Material category '{category}' not found. Available: {self.get_categories()}")
        return self.db[category]

    def get_material(self, category: str, key: str) -> Dict[str, Any]:
        cat_dict = self.get_materials_by_category(category)
        if key not in cat_dict:
            raise KeyError(f"Material key '{key}' not found under category '{category}'. Available: {list(cat_dict.keys())}")
        return cat_dict[key]

    def filter_materials(self, category: str, max_k: Optional[float] = None, max_cost: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Filter materials in a category by maximum thermal conductivity or cost."""
        cat_dict = self.get_materials_by_category(category)
        filtered = {}
        for k, props in cat_dict.items():
            if max_k is not None and "k" in props and props["k"] > max_k:
                continue
            if max_cost is not None and "cost_per_m2_base" in props and props["cost_per_m2_base"] > max_cost:
                continue
            filtered[k] = props
        return filtered

# Export helper lists for legacy code compatibility
MATERIALS = MATERIAL_DATABASE["wall"]
INSULATION_MATERIALS = MATERIAL_DATABASE["insulation"]
GLAZING = MATERIAL_DATABASE["glazing"]
PCM = MATERIAL_DATABASE["pcm"]

WALL_MATERIAL_LIST = list(MATERIALS.keys())
INSULATION_LIST = list(INSULATION_MATERIALS.keys())
GLAZING_LIST = list(GLAZING.keys())

if __name__ == "__main__":
    mdb = MaterialDatabase()
    print("Wall Materials:", list(mdb.get_materials_by_category("wall").keys()))
    print("Filter k <= 0.5:", list(mdb.filter_materials("wall", max_k=0.55).keys()))
