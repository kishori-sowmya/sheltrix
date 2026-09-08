# AI-Assisted Shelter Thermal Design Platform (SIH MVP)

Surrogate-assisted design exploration for climate-responsive shelters, with
physics-model grounding and a scaffold for ANSYS validation.

**Pipeline:** Climate + Materials → RC-Network Physics Model (Tier-1) / ANSYS
(Tier-2) → Simulation Dataset → XGBoost Surrogate → NSGA-II Optimizer →
Validation → Dashboard

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline (in order)

```bash
# Stage 2: generate the Tier-1 dataset (LHS sampling x RC-network physics model)
python3 src/generate_dataset.py

# Stage 3: train 8 XGBoost surrogate models, print MAE/RMSE/R2/MAPE
python3 src/train_surrogate.py

# Stage 3.5: sanity-check feature importance against physical intuition
python3 src/feature_importance.py

# Stage 4: NSGA-II optimizer searches thousands of designs via the surrogate
python3 src/optimizer.py

# Stage 5: validate top designs (surrogate vs re-run physics model;
# ANSYS column is a clearly-labeled placeholder - see note below)
python3 src/validate_against_ansys.py

# Stage 6: launch the dashboard (ties all of the above together)
streamlit run dashboard.py
```

## What's real vs. what's a placeholder — read this before presenting

This MVP is built to be **honest under judge scrutiny**, not to overclaim.

| Component | Status |
|---|---|
| Climate profiles (Ladakh winter/summer) | Synthetic but physically-reasonable diurnal curves. **Replace with real NASA POWER / IMD data for final submission** (`data/climate_ladakh.py` has a note on this). |
| Material properties | Standard reference-range values. Cite exact datasheet/ASHRAE sources before final submission. |
| RC-network thermal model (Tier-1) | Fully implemented, runs in ~0.1ms/sample. This is a legitimate simplified building-physics method (lumped capacitance), not a stand-in claimed to be ANSYS. |
| Dataset (600 rows) | Real output of the Tier-1 model over LHS-sampled design space. No fabricated numbers. |
| XGBoost surrogate (8 models) | Actually trained; metrics in `models/surrogate_eval_metrics.csv` are real, not invented. R² ranges ~0.90–0.98 across outputs, with two flagged caveats (see dashboard Tab 2 caption). |
| NSGA-II optimizer | Actually runs; genuinely evaluates 4,000 surrogate predictions in ~0.3 seconds. This is your strongest, most honest "why AI helps" number. |
| **ANSYS validation** | **NOT connected to real ANSYS in this environment** (no license/solver access here). `validate_against_ansys.py` re-runs top designs through the Tier-1 physics model directly (a genuine surrogate-vs-physics check, ~8% mean error) and includes a clearly-labeled `higher_fidelity_stub()` placeholder with synthetic perturbation, meant only to demonstrate the reporting format. **Before your SIH demo, replace this stub with real ANSYS runs on the 30 rows already flagged** `ansys_validation_candidate=True` **in `data/dataset_v1.csv`.** |
| Dashboard | Fully functional Streamlit app wired to real outputs from all stages above. |

## Project structure

```
shelter_ai/
├── data/
│   ├── climate_ladakh.py       # Stage 1: climate profiles
│   ├── materials.py            # Stage 1: material database
│   └── dataset_v1.csv          # Stage 2: generated dataset (600 rows)
├── src/
│   ├── thermal_model.py        # Stage 1/2: RC-network physics engine
│   ├── sampler.py               # Stage 2: Latin Hypercube sampling
│   ├── generate_dataset.py      # Stage 2: pipeline
│   ├── train_surrogate.py       # Stage 3: XGBoost training + metrics
│   ├── feature_importance.py    # Stage 3: explainability sanity check
│   ├── optimizer.py             # Stage 4: NSGA-II via pymoo
│   └── validate_against_ansys.py # Stage 5: validation (ANSYS stub)
├── models/                       # Saved XGBoost models + metrics
├── outputs/                      # Pareto front + validation reports
├── dashboard.py                  # Stage 6: Streamlit UI
└── requirements.txt
```

## Immediate next steps for the real submission

1. Swap synthetic climate profiles for real NASA POWER data (Leh: 34.15°N, 77.58°E).
2. Run the 30 flagged Tier-2 rows through actual ANSYS; replace `higher_fidelity_stub()`.
3. Extend the optimizer to search categorical variables (materials) jointly, not just fixed.
4. Add uncertainty-aware surrogate (GPR) as the "advanced tier" differentiator.
