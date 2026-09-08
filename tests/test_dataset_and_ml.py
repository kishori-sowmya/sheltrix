"""
Module 17: Automated Tests for Dataset Pipeline, Surrogate ML, and Quality Control
"""

import pytest
import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.dataset_generator import sample_parameter_space
from src.ml.quality_control import SurrogateQualityControl
from src.optimization.optimizer import FeasibilityChecker


def test_lhs_parameter_sampling():
    samples = sample_parameter_space(n_samples=50, seed=42)
    assert len(samples) == 50
    for s in samples:
        assert 3.0 <= s["length"] <= 8.0
        assert 3.0 <= s["width"] <= 6.0
        assert 0.05 <= s["window_ratio"] <= 0.40


def test_quality_control_bounds():
    qc = SurrogateQualityControl()
    valid, warnings = qc.validate_input_bounds({"length": 5.0, "width": 4.0, "insulation_thickness_mm": 80.0})
    assert valid, "Valid parameters should pass without warnings"

    invalid, warnings2 = qc.validate_input_bounds({"length": 25.0, "width": 4.0})
    assert not invalid, "Out of bounds parameter must fail validation"
    assert len(warnings2) > 0


def test_feasibility_checker():
    feasible_design = {"length": 5.0, "width": 4.0, "window_ratio": 0.15, "insulation_thickness_mm": 80.0}
    is_feas, reasons = FeasibilityChecker.is_feasible(feasible_design, max_footprint=30.0)
    assert is_feas, f"Feasible design failed: {reasons}"

    infeasible_design = {"length": 10.0, "width": 10.0, "window_ratio": 0.50, "insulation_thickness_mm": 0.0}
    is_feas2, reasons2 = FeasibilityChecker.is_feasible(infeasible_design, max_footprint=30.0)
    assert not is_feas2, "Infeasible design must be rejected"
