"""
Latin Hypercube Sampling over the shelter design parameter space (Stage 2.1).
Produces a table of shelter configurations to be run through the thermal
model (Tier-1) and, for a stratified subset, through ANSYS (Tier-2).
"""
import numpy as np
from scipy.stats import qmc
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.materials import WALL_MATERIAL_LIST, INSULATION_LIST, GLAZING_LIST

# Continuous parameter bounds, per Stage 1.4
CONTINUOUS_BOUNDS = {
    "length": (3.0, 8.0),
    "width": (3.0, 6.0),
    "wall_height": (2.2, 3.5),
    "roof_angle": (0.0, 45.0),
    "orientation_deg": (0.0, 180.0),   # 0=south-facing (max winter sun), 180=north
    "insulation_thickness_mm": (0.0, 150.0),
    "window_ratio": (0.05, 0.40),
    "pcm_thickness_mm": (0.0, 50.0),
}
CONT_KEYS = list(CONTINUOUS_BOUNDS.keys())


def generate_samples(n_samples=250, seed=42):
    sampler = qmc.LatinHypercube(d=len(CONT_KEYS), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    lower = np.array([CONTINUOUS_BOUNDS[k][0] for k in CONT_KEYS])
    upper = np.array([CONTINUOUS_BOUNDS[k][1] for k in CONT_KEYS])
    scaled = qmc.scale(unit_samples, lower, upper)

    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_samples):
        row = dict(zip(CONT_KEYS, scaled[i]))
        row["wall_material"] = rng.choice(WALL_MATERIAL_LIST)
        row["roof_material"] = rng.choice(WALL_MATERIAL_LIST)
        row["floor_material"] = rng.choice(["concrete", "brick", "cseb"])
        row["insulation_material"] = rng.choice(INSULATION_LIST)
        row["glazing_type"] = rng.choice(GLAZING_LIST)
        row["pcm_present"] = int(rng.random() < 0.4)  # 40% of samples include PCM
        if row["pcm_present"] == 0:
            row["pcm_thickness_mm"] = 0.0
        row["sample_id"] = i
        configs.append(row)
    return configs


if __name__ == "__main__":
    samples = generate_samples(n_samples=5)
    for s in samples:
        print(s)
