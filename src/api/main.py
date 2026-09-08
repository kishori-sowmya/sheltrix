"""
Module 14: FastAPI Backend Server
Exposes REST endpoints for climate profiles, materials, thermal prediction,
multi-objective optimization, physics simulation, ANSYS adapter, and surrogate metrics.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import uvicorn, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_modules.climate_manager import ClimateManager
from src.data_modules.material_db import MaterialDatabase
from src.physics.rc_model import simulate_shelter as run_rc_sim
from src.physics.ansys_adapter import get_simulation_engine
from src.ml.surrogate_trainer import load_data_and_preprocess
from src.ml.quality_control import SurrogateQualityControl
from src.optimization.optimizer import run_optimization, build_surrogate_feature_vector, load_surrogate_models
from src.explainability.explainer import DesignExplainer

app = FastAPI(
    title="AI-Assisted Shelter Thermal Design API",
    description="REST API for area-specific shelter thermal modeling, surrogate predictions, optimization, and validation.",
    version="1.0.0",
)

cm = ClimateManager()
mdb = MaterialDatabase()
qc = SurrogateQualityControl()
explainer = DesignExplainer()


class DesignParametersInput(BaseModel):
    length: float = Field(5.0, ge=3.0, le=8.0, description="Shelter length in meters")
    width: float = Field(4.0, ge=3.0, le=6.0, description="Shelter width in meters")
    wall_height: float = Field(2.8, ge=2.2, le=3.5, description="Wall height in meters")
    roof_angle: float = Field(20.0, ge=0.0, le=45.0, description="Roof angle in degrees")
    orientation_deg: float = Field(0.0, ge=0.0, le=180.0, description="Solar orientation (0=South)")
    insulation_thickness_mm: float = Field(80.0, ge=0.0, le=150.0, description="Insulation thickness in mm")
    window_ratio: float = Field(0.15, ge=0.05, le=0.40, description="Window-to-wall ratio")
    pcm_thickness_mm: float = Field(20.0, ge=0.0, le=50.0, description="PCM thickness in mm")
    wall_material: str = Field("cseb", description="Wall material key")
    roof_material: str = Field("sandwich_panel", description="Roof material key")
    floor_material: str = Field("concrete", description="Floor material key")
    insulation_material: str = Field("xps_foam", description="Insulation material key")
    glazing_type: str = Field("double_pane", description="Glazing type key")
    pcm_present: int = Field(1, ge=0, le=1, description="PCM presence binary flag")


class OptimizationRequest(BaseModel):
    climate_scenario: str = Field("winter", description="Climate scenario name")
    pop_size: int = Field(100, ge=20, le=300)
    n_gen: int = Field(40, ge=5, le=100)
    max_footprint: float = Field(48.0, ge=9.0, le=100.0)


@app.get("/health")
def health_check():
    surr_ready, surr_msg = qc.is_surrogate_ready()
    ansys_engine = get_simulation_engine("ansys")
    return {
        "status": "HEALTHY",
        "surrogate_model_ready": surr_ready,
        "surrogate_message": surr_msg,
        "ansys_solver_available": ansys_engine.is_available(),
        "ansys_status_notice": "ANSYS unavailable — using Tier-1 physics validation." if not ansys_engine.is_available() else "ANSYS active",
    }


@app.get("/climates")
def get_climates():
    names = cm.get_climate_names()
    summaries = {name: cm.get_summary(name) for name in names}
    return {"available_climates": names, "summaries": summaries}


@app.get("/materials")
def get_materials(category: Optional[str] = None):
    if category:
        try:
            return mdb.get_materials_by_category(category)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {cat: mdb.get_materials_by_category(cat) for cat in mdb.get_categories()}


@app.post("/design/predict")
def predict_thermal_performance(design: DesignParametersInput, climate_scenario: str = "winter"):
    # 1. Quality Control check
    valid, warnings = qc.validate_input_bounds(design.dict())

    # 2. Run fast XGBoost surrogate prediction
    feature_names, heat_m, comf_m, heat_loss_m = load_surrogate_models()
    feat_vec = build_surrogate_feature_vector(
        [design.length, design.width, design.wall_height, design.roof_angle, design.orientation_deg,
         design.insulation_thickness_mm, design.window_ratio, design.pcm_thickness_mm],
        feature_names, climate_scenario, design.dict()
    )

    heating_kWh = qc.sanitize_prediction(float(heat_m.predict([feat_vec])[0]), "heating_energy_required_kWh")
    comfort_hrs = qc.sanitize_prediction(float(comf_m.predict([feat_vec])[0]), "comfort_hours")
    heat_loss_kWh = qc.sanitize_prediction(float(heat_loss_m.predict([feat_vec])[0]), "total_heat_loss_kWh")

    explanation = explainer.explain_design_recommendation(
        {**design.dict(), "heating_energy_required_kWh": heating_kWh}, target="heating_energy_required_kWh"
    )

    return {
        "climate_scenario": climate_scenario,
        "predictions": {
            "heating_energy_required_kWh": round(heating_kWh, 2),
            "comfort_hours": round(comfort_hrs, 1),
            "total_heat_loss_kWh": round(heat_loss_kWh, 2),
        },
        "warnings": warnings,
        "explanation": explanation,
    }


@app.post("/simulation/rc")
def run_rc_simulation(design: DesignParametersInput, climate_scenario: str = "winter"):
    try:
        res = run_rc_sim(design.dict(), climate_scenario=climate_scenario)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/ansys")
def run_ansys_simulation(design: DesignParametersInput, climate_scenario: str = "winter"):
    ansys_engine = get_simulation_engine("ansys")
    return ansys_engine.run_simulation(design.dict(), climate_scenario=climate_scenario)


@app.post("/optimization/run")
def run_optimization_endpoint(req: OptimizationRequest):
    try:
        pareto_df, elapsed, total_evals, candidates = run_optimization(
            climate_scenario=req.climate_scenario,
            pop_size=req.pop_size,
            n_gen=req.n_gen,
            max_footprint=req.max_footprint,
        )
        return {
            "climate_scenario": req.climate_scenario,
            "total_evaluations": total_evals,
            "elapsed_seconds": round(elapsed, 2),
            "pareto_design_count": len(pareto_df),
            "top_candidates": candidates,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/surrogate/metrics")
def get_surrogate_metrics():
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "v1.0", "surrogate_eval_metrics.csv")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "surrogate_eval_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return df.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="Surrogate metrics not found.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
