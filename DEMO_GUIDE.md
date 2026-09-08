# SIH Hackathon Demo Guide & Step-by-Step Presentation Script

Follow this 14-step flow to deliver a seamless, high-impact SIH presentation demonstrating the complete AI Surrogate + Physics + Optimization + Validation pipeline.

---

## 14-Step Demonstration Flow

```text
Step 1: Open Dashboard
   ↓ `streamlit run dashboard.py`
Step 2: Navigate to Problem & Solution Tab
   ↓ Explain the core innovation: "AI Surrogate for Exploration + Physics for Validation"
Step 3: Select Region
   ↓ Select "Ladakh (High-Altitude Cold Desert)"
Step 4: Select Climate Scenario
   ↓ Select "Ladakh Severe Winter (Jan)" (-18°C to -2°C)
Step 5: Define Shelter Geometry
   ↓ Set Length: 5.0m, Width: 4.0m, Wall Height: 2.8m, Roof Angle: 20°
Step 6: Select Materials & Insulation
   ↓ Select CSEB wall, Sandwich Roof, XPS Insulation (80mm), Double Glazing, PCM (20mm)
Step 7: View 3D Digital Twin Model
   ↓ Open "Design Studio & 3D" tab to inspect 3D mesh, window cutout, and solar orientation vector
Step 8: Run Instant Prediction
   ↓ Click "Run Instant Prediction" -> Show <1ms inference vs hours in ANSYS
Step 9: Review Diurnal Temperature Curve
   ↓ Inspect 24-hour temperature profile vs 18°C comfort threshold
Step 10: Run NSGA-II Multi-Objective Optimization
   ↓ Click "Search Pareto Optimal Shelter Designs" -> Evaluates 4,000 candidates in ~0.3s
Step 11: Explore Pareto Trade-Off Front
   ↓ Show Pareto scatter plot (Heating Energy vs Comfort Hours vs Insulation Thickness)
Step 12: Side-by-Side Design Comparison
   ↓ Inspect Comparison Matrix contrasting User Design vs High-Insulation Solar Design
Step 13: Explain Recommendation Driver (XAI)
   ↓ View SHAP/Feature Importance breakdown & natural language scientific explanation
Step 14: Tiered Validation Studio
   ↓ Show Surrogate vs Physics vs ANSYS Status ("ANSYS unavailable — using physics-model validation")
```

---

## Quick Execution Commands

```bash
# 1. Generate physics simulation dataset (600 rows across climate scenarios)
python src/ml/dataset_generator.py

# 2. Train XGBoost surrogate models & save versioned artifacts
python src/ml/surrogate_trainer.py

# 3. Run NSGA-II multi-objective optimization
python src/optimization/optimizer.py

# 4. Run automated test suite
python -m pytest tests/

# 5. Launch FastAPI backend REST API
python src/api/main.py

# 6. Launch Streamlit Dashboard
streamlit run dashboard.py
```
