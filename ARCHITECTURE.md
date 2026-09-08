# System Architecture — AI-Assisted Shelter Thermal Design Platform

## Architecture Overview

```text
USER / DASHBOARD (Streamlit) / REST API (FastAPI)
                       ↓
  Climate Data Manager (Module 1) + Material Database (Module 2)
                       ↓
  Design Parameter Config (`config/design_parameters.yaml`) (Module 3)
                       ↓
  Latin Hypercube Sampler (Module 5)
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                    SIMULATION ENGINE (Module 6)               │
│  ├── RCPhysicsEngine (Tier-1 Lumped Capacitance) [Module 4]  │
│  └── ANSYSEngineAdapter (Tier-2 CFD Integration) [Module 6]  │
└──────────────────────────────────────────────────────────────┘
                       ↓
  Training Dataset & Provenance Metadata (`datasets/`) [Module 5]
                       ↓
  XGBoost Surrogate Models & Quality Control [Modules 7, 8, 16]
                       ↓
  NSGA-II Multi-Objective Optimizer & Feasibility [Modules 9, 10]
                       ↓
  Explainable AI (SHAP / Feature Sensitivity) [Module 11]
                       ↓
  Plotly 3D Digital Twin Visualizer [Module 13]
                       ↓
  SQLite Database (`database/shelter.db`) [Module 15]
```

## Core Modules & Design Responsibilities

1. **`src/data_modules/climate_manager.py`**: Serves Ladakh winter/summer/shoulder profiles, handles user CSV uploads, imputes missing values, and validates unit ranges.
2. **`src/data_modules/material_db.py`**: Manages thermal conductivity, density, specific heat, absorptivity, emissivity, and cost properties cited from ASHRAE / ISO / IS standards.
3. **`src/physics/rc_model.py`**: 1-zone lumped capacitance heat balance model with 3-day warm-up spinup, floor thermal mass, and PCM latent heat enthalpy buffering.
4. **`src/physics/ansys_adapter.py`**: Abstract `SimulationEngine` interface providing APDL/Fluent setup script generators, status monitoring, and explicit offline status notice ("ANSYS unavailable — using physics validation").
5. **`src/ml/surrogate_trainer.py`**: Trains versioned XGBoost regressors on Tier-1 dataset (excluding Tier-2 validation candidates) with 5-fold cross-validation.
6. **`src/ml/quality_control.py`**: Pre-validates parameter bounds and surrogate $R^2$ metrics before optimization.
7. **`src/optimization/optimizer.py`**: Performs NSGA-II multi-objective optimization across continuous parameters and material selections; enforces engineering feasibility.
8. **`src/explainability/explainer.py`**: Quantifies local feature contributions with scientific disclaimers.
9. **`src/visualization/shelter_3d.py`**: Generates interactive 3D Plotly mesh renderings of shelter geometry, roof angle, window cutouts, and solar orientation.
10. **`src/api/main.py`**: REST API built with FastAPI.
11. **`dashboard.py`**: Streamlit dashboard presenting all 7 core views.
