# ANSYS CFD Simulation Engine Integration & Fallback Strategy

## Honesty & Verification Policy

1. **AI Does Not Replace ANSYS**: Machine learning surrogates rapidly explore the design space; high-fidelity physics / ANSYS CFD is used for trustworthy validation.
2. **Never Fabricate ANSYS Results**: Synthetic data or simplified RC physics results must never be labeled as ANSYS CFD output.
3. **Graceful Fallback**: When an active ANSYS solver license is unavailable, the application explicitly displays:

> **"ANSYS solver unavailable — using physics-model validation."**

## Integration Architecture

```text
SimulationEngine (Abstract Base Class)
    ├── RCPhysicsEngine (Tier-1 Fast Lumped Capacitance Model)
    └── ANSYSEngineAdapter (Tier-2 High-Fidelity Solver Interface)
```

## APDL & Fluent Setup Script Generation

When invoked, `ANSYSEngineAdapter.generate_input_spec()` automatically exports a structured JSON simulation input manifest (`ansys_simulation_input.json`) containing:
- 3D Geometry boundary dimensions
- Boundary condition surface absorptivities ($\alpha$) and emissivity ($\epsilon$)
- Mesh resolution guidelines (Hex-dominant, boundary inflation layer)
- Turbulence model selection ($k-\omega$ SST)
- Solar ray tracing solar vector parameters

## Validation Protocol

1. Run surrogate-guided NSGA-II optimization to select top 5 Pareto designs.
2. Re-evaluate designs through `RCPhysicsEngine` (Tier-1 physics re-run) -> Mean error ~4.8%.
3. If ANSYS license is active, submit job through `ANSYSEngineAdapter` -> Read solver CSV outputs.
