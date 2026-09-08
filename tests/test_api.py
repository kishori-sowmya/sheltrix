"""
Module 17: Automated Tests for FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "surrogate_model_ready" in data


def test_api_climates_endpoint():
    response = client.get("/climates")
    assert response.status_code == 200
    data = response.json()
    assert "available_climates" in data
    assert "winter" in data["available_climates"]


def test_api_materials_endpoint():
    response = client.get("/materials")
    assert response.status_code == 200
    data = response.json()
    assert "wall" in data
    assert "cseb" in data["wall"]


def test_api_predict_endpoint():
    payload = {
        "length": 5.0, "width": 4.0, "wall_height": 2.8, "roof_angle": 20.0,
        "orientation_deg": 0.0, "insulation_thickness_mm": 80.0, "window_ratio": 0.15,
        "pcm_thickness_mm": 20.0, "wall_material": "cseb", "roof_material": "sandwich_panel",
        "floor_material": "concrete", "insulation_material": "xps_foam",
        "glazing_type": "double_pane", "pcm_present": 1
    }
    response = client.post("/design/predict?climate_scenario=winter", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "heating_energy_required_kWh" in data["predictions"]
