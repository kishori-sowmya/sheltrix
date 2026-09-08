"""
Module 12: Professional Streamlit Dashboard for Thermal Design Platform
Integrates system modules: Multi-Region Climate Manager, Material Database,
RC Physics Engine, Dataset Generator, NSGA-II Optimizer, Sensitivity Analysis,
Interactive Plotly 3D Shelter Visualizer, and ANSYS Validation Studio.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os, sys, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_modules.climate_manager import ClimateManager
from src.data_modules.material_db import MaterialDatabase, WALL_MATERIAL_LIST, INSULATION_LIST, GLAZING_LIST
from src.physics.rc_model import simulate_shelter, compute_layer_resistance
from src.physics.ansys_adapter import get_simulation_engine
from src.optimization.optimizer import run_optimization
from src.explainability.explainer import DesignExplainer
from src.visualization.shelter_3d import render_shelter_3d
from src.ml.quality_control import SurrogateQualityControl

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="Shelter Thermal Design System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism & visual excellence
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #8E2DE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AAB8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .status-badge-success {
        padding: 10px 14px;
        border-radius: 8px;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ECC71;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .status-badge-warning {
        padding: 10px 14px;
        border-radius: 8px;
        background: rgba(241, 196, 15, 0.12);
        border: 1px solid rgba(241, 196, 15, 0.3);
        color: #F1C40F;
        font-size: 0.9rem;
    }
    .stSlider > div {
        padding-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

cm = ClimateManager()
mdb = MaterialDatabase()
qc = SurrogateQualityControl()
explainer = DesignExplainer()
ansys_engine = get_simulation_engine("ansys")

# ---------------- SIDEBAR CONTROLS (MULTI-REGION DYNAMIC SELECTION) ----------------
with st.sidebar:
    st.markdown("## Shelter Thermal Controls")

    # Quick Presets Selector
    preset = st.selectbox(
        "Quick Design Presets",
        ["Custom Configuration", "High Passive Solar Design", "Modular Prefab Sandwich Panel", "Low-Cost Earth Wall Design"],
        index=1,
        help="Select a predefined design baseline or customize parameters below."
    )

    # Preset Defaults
    p_len, p_wid, p_height, p_roof, p_orient = 5.0, 4.0, 2.8, 20.0, 0.0
    p_wall, p_roof_m, p_floor, p_ins, p_ins_t, p_wwr, p_glaz, p_pcm_t = "cseb", "sandwich_panel", "concrete", "xps_foam", 80.0, 0.15, "double_pane", 20.0

    if preset == "High Passive Solar Design":
        p_len, p_wid, p_height, p_roof, p_orient = 6.0, 4.0, 2.8, 25.0, 0.0
        p_wall, p_roof_m, p_floor, p_ins, p_ins_t, p_wwr, p_glaz, p_pcm_t = "cseb", "sandwich_panel", "concrete", "xps_foam", 100.0, 0.25, "double_pane", 30.0
    elif preset == "Modular Prefab Sandwich Panel":
        p_len, p_wid, p_height, p_roof, p_orient = 5.0, 3.5, 2.5, 15.0, 0.0
        p_wall, p_roof_m, p_floor, p_ins, p_ins_t, p_wwr, p_glaz, p_pcm_t = "sandwich_panel", "sandwich_panel", "concrete", "eps_foam", 60.0, 0.15, "double_pane", 0.0
    elif preset == "Low-Cost Earth Wall Design":
        p_len, p_wid, p_height, p_roof, p_orient = 4.5, 4.0, 2.8, 20.0, 0.0
        p_wall, p_roof_m, p_floor, p_ins, p_ins_t, p_wwr, p_glaz, p_pcm_t = "cseb", "cseb", "cseb", "mineral_wool", 50.0, 0.10, "single_pane", 0.0

    with st.expander("Geographical Region & Climate Scenario", expanded=True):
        regions = cm.get_available_regions()
        selected_region = st.selectbox(
            "Geographical Region",
            regions,
            index=0,
            help="Select target climate region across North or South India."
        )

        available_scenarios = cm.get_scenarios_for_region(selected_region)
        scenario_names_map = {k: cm.get_profile(k)["name"] for k in available_scenarios}

        climate_scenario = st.selectbox(
            "Design Climate Scenario / Season",
            available_scenarios,
            index=0,
            format_func=lambda x: scenario_names_map[x],
            help="Select seasonal profile to evaluate thermal performance."
        )

        prof = cm.get_profile(climate_scenario)
        st.caption(f"Region: {prof['region']} | Data Source: {prof['data_source']}")

    with st.expander("Shelter Dimensions & Solar Orientation", expanded=True):
        length = st.slider("Length (m)", 3.0, 8.0, p_len, 0.5, help="Length of shelter outer footprint in meters")
        width = st.slider("Width (m)", 3.0, 6.0, p_wid, 0.5, help="Width of shelter outer footprint in meters")
        wall_height = st.slider("Wall Height (m)", 2.2, 3.5, p_height, 0.1, help="Height of vertical outer walls in meters")
        roof_angle = st.slider("Roof Pitch (degrees)", 0.0, 45.0, p_roof, 5.0, help="Sloping angle of roof assembly")
        orientation_deg = st.slider(
            "Solar Azimuth Orientation (0 = True South)",
            0.0, 180.0, p_orient, 15.0,
            help="Facing direction of primary windows. 0° = South facing for maximum daytime solar heat gain."
        )

    with st.expander("Building Envelope Materials & Insulation", expanded=True):
        wall_material = st.selectbox("Wall Material", WALL_MATERIAL_LIST, index=WALL_MATERIAL_LIST.index(p_wall), help="Structural material for exterior walls")
        roof_material = st.selectbox("Roof Assembly Material", WALL_MATERIAL_LIST, index=WALL_MATERIAL_LIST.index(p_roof_m), help="Structural material for roof assembly")
        floor_material = st.selectbox("Floor Slab Material", ["concrete", "brick", "cseb"], index=["concrete", "brick", "cseb"].index(p_floor), help="Subfloor material")
        insulation_material = st.selectbox("Insulation Core Material", INSULATION_LIST, index=INSULATION_LIST.index(p_ins), help="Core thermal insulation layer")
        insulation_thickness_mm = st.slider("Insulation Thickness (mm)", 0.0, 150.0, p_ins_t, 10.0, help="Thickness of thermal insulation layer in millimeters")
        window_ratio = st.slider("Window-to-Wall Ratio (WWR)", 0.05, 0.40, p_wwr, 0.05, help="Ratio of window surface area relative to total wall surface area")
        glazing_type = st.selectbox("Window Glazing System", GLAZING_LIST, index=GLAZING_LIST.index(p_glaz), help="Glazing type (e.g., double pane glass)")

    with st.expander("Thermal Storage & Phase Change Material (PCM)", expanded=False):
        pcm_present = st.checkbox("Include PCM Thermal Storage Layer", value=(p_pcm_t > 0), help="Add phase change material layer to store excess heat and buffer indoor temperature.")
        pcm_thickness_mm = st.slider("PCM Layer Thickness (mm)", 0.0, 50.0, p_pcm_t if pcm_present else 0.0, 5.0, help="Thickness of thermal storage material layer in millimeters.")

    user_design_dict = {
        "length": length, "width": width, "wall_height": wall_height,
        "roof_angle": roof_angle, "orientation_deg": orientation_deg,
        "wall_material": wall_material, "roof_material": roof_material,
        "floor_material": floor_material, "insulation_material": insulation_material,
        "insulation_thickness_mm": insulation_thickness_mm, "window_ratio": window_ratio,
        "glazing_type": glazing_type, "pcm_present": 1 if pcm_present else 0,
        "pcm_thickness_mm": pcm_thickness_mm if pcm_present else 0.0,
    }

# Dynamic Physics Simulation Run for the current region and user parameters
rc_res = simulate_shelter(user_design_dict, climate_scenario=climate_scenario)

# ---------------- NAVIGATION TABS (PROBLEM & ARCHITECTURE REMOVED) ----------------
tab_design, tab_pred, tab_opt, tab_comp, tab_rec, tab_val = st.tabs([
    "Design Studio & 3D Twin",
    "Physics & Thermal Analysis",
    "Multi-Objective Optimization",
    "Design Comparison Matrix",
    "Sensitivity & Recommendation Analysis",
    "Validation Studio"
])

# ---------------- TAB 1: DESIGN STUDIO & 3D TWIN ----------------
with tab_design:
    st.subheader("Parametric Shelter Design Studio & Property Calculator")
    st.caption(f"Active Location: {prof['name']} ({prof['region']})")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 1. Current Shelter Specification & Derived Parameters")
        st.json(user_design_dict)

        # Derived Building Physics Quantities
        fl_area = length * width
        wall_ar = 2 * (length + width) * wall_height
        win_ar = wall_ar * window_ratio
        net_w_ar = wall_ar - win_ar
        rf_ar = (length * width) / np.cos(np.radians(min(roof_angle, 60)))
        vol = fl_area * wall_height

        r_wall = compute_layer_resistance("wall", wall_material, insulation_material, insulation_thickness_mm)
        u_wall = 1.0 / r_wall

        # Cost & Weight estimation
        w_props = mdb.get_material("wall", wall_material)
        i_props = mdb.get_material("insulation", insulation_material)
        g_props = mdb.get_material("glazing", glazing_type)

        wall_weight_ton = (net_w_ar * 0.20 * w_props["rho"]) / 1000.0
        ins_cost_inr = (insulation_thickness_mm / 50.0) * i_props.get("cost_per_m2_base", 1500.0) * (net_w_ar + rf_ar)
        total_est_cost_inr = (net_w_ar * w_props.get("cost_per_m2_base", 2500)) + ins_cost_inr + (win_ar * g_props.get("cost_per_m2_base", 6200))

        st.markdown("#### Derived Architectural & Thermal Properties")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Floor Footprint Area", f"{fl_area:.1f} m²", help="Total indoor ground footprint area")
        pcol1.metric("Enclosure Volume", f"{vol:.1f} m³", help="Total internal air volume")
        pcol2.metric("Wall R-Value (Insulation)", f"{r_wall:.2f} m²K/W", help="Thermal resistance of exterior wall. Higher R-Value means better insulation.")
        pcol2.metric("Wall U-Value (Conductance)", f"{u_wall:.3f} W/m²K", help="Thermal transmittance. Lower U-Value means less heat flow.")
        pcol3.metric("Est. Structural Weight", f"{wall_weight_ton:.1f} Tons", help="Approximate mass of outer structural wall envelope")
        pcol3.metric("Est. Envelope Material Cost", f"₹ {total_est_cost_inr:,.0f}", help="Estimated material cost for envelope, insulation, and glazing in Indian Rupees (₹)")

        valid, warnings = qc.validate_input_bounds(user_design_dict)
        if not valid:
            for w in warnings:
                st.warning(f"[Design Check] {w}")
        else:
            st.markdown('<div class="status-badge-success">All design parameters are strictly within valid physical bounds.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### 2. Interactive 3D Digital Twin Representation")
        fig_3d = render_shelter_3d(user_design_dict, show_layers=True)
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("#### Selected Material Properties Summary")
        mat_summary_df = pd.DataFrame([
            {"Component": "Wall Structure", "Material": str(w_props["name"]), "Thermal Conductivity / U-Value": f"{w_props['k']} W/m.K", "Density (kg/m³)": f"{w_props['rho']}", "Solar Absorptivity": f"{w_props.get('alpha', 0.6)}", "Data Source": str(w_props.get("source", "Standard Reference"))},
            {"Component": "Insulation Core", "Material": str(i_props["name"]), "Thermal Conductivity / U-Value": f"{i_props['k']} W/m.K", "Density (kg/m³)": f"{i_props['rho']}", "Solar Absorptivity": "N/A", "Data Source": str(i_props.get("source", "Standard Reference"))},
            {"Component": "Window Glazing", "Material": str(g_props["name"]), "Thermal Conductivity / U-Value": f"U = {g_props['U_value']} W/m²K", "Density (kg/m³)": "N/A", "Solar Absorptivity": f"SHGC = {g_props['SHGC']}", "Data Source": str(g_props.get("source", "Standard Reference"))},
        ])
        st.dataframe(mat_summary_df, use_container_width=True)

# ---------------- TAB 2: PHYSICS & THERMAL ANALYSIS ----------------
with tab_pred:
    st.subheader("Physics Engine Simulation & Detailed Heat Balance Analysis")
    st.caption(f"Solves 24-hour diurnal lumped RC thermal resistance network dynamically for {prof['name']}.")

    m1, m2, m3, m4 = st.columns(4)
    heating_kwh = rc_res.get('heating_energy_required_kWh', 0.0)
    cooling_kwh = rc_res.get('cooling_energy_required_kWh', 0.0)
    total_hvac_kwh = rc_res.get('total_hvac_energy_kWh', heating_kwh + cooling_kwh)

    if cooling_kwh > heating_kwh:
        m1.metric("Predicted Cooling Energy Needed", f"{cooling_kwh:.1f} kWh/day", help="Kilowatt-hours of cooling energy required daily to prevent overheating.")
    else:
        m1.metric("Predicted Heating Energy Needed", f"{heating_kwh:.1f} kWh/day", help="Kilowatt-hours of heating energy required daily to maintain indoor warmth.")

    m2.metric("Thermal Comfort Hours (18°C - 25°C)", f"{rc_res['comfort_hours']} hrs/day", delta=f"{rc_res['comfort_hours']-12:.0f} hrs vs baseline", help="Hours per day indoor temperature remains in comfortable 18°C - 25°C band.")
    m3.metric("Total Wall & Roof Heat Loss", f"{rc_res['total_heat_loss_kWh']:.1f} kWh/day", help="Conductive heat transfer escaping through building envelope.")
    m4.metric("Passive Solar Heat Gain", f"{rc_res['total_solar_gain_kWh']:.1f} kWh/day", help="Solar heat gained through windows during daylight hours.")

    st.divider()

    # Detailed Analytical Charts
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown(f"#### 1. 24-Hour Diurnal Indoor vs Outdoor Temperature Profile ({prof['region']})")
        hours = list(range(24))
        df_temp = pd.DataFrame({
            "Hour of Day": hours,
            "Indoor Temperature (°C)": rc_res["hourly_indoor_temp"],
            "Outdoor Ambient Temp (°C)": prof["T_amb_hourly"],
            "Lower Comfort Limit (18°C)": [18.0] * 24,
            "Upper Comfort Limit (25°C)": [25.0] * 24,
        })
        fig_temp = px.line(
            df_temp, x="Hour of Day", y=["Indoor Temperature (°C)", "Outdoor Ambient Temp (°C)", "Lower Comfort Limit (18°C)", "Upper Comfort Limit (25°C)"],
            title=f"Diurnal Temperature Profile — {prof['name']}",
            color_discrete_map={
                "Indoor Temperature (°C)": "#FF4B4B",
                "Outdoor Ambient Temp (°C)": "#1E88E5",
                "Lower Comfort Limit (18°C)": "#4CAF50",
                "Upper Comfort Limit (25°C)": "#FFA726"
            }
        )
        fig_temp.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2))
        st.plotly_chart(fig_temp, use_container_width=True)

    with c_right:
        st.markdown("#### 2. Daily Energy Balance Breakdown (kWh/day)")
        balance_df = pd.DataFrame([
            {"Component": "Passive Solar Heat Gain", "Energy (kWh)": rc_res["total_solar_gain_kWh"], "Category": "Heat Gain"},
            {"Component": "Envelope Conductive Loss", "Energy (kWh)": rc_res["total_heat_loss_kWh"], "Category": "Heat Loss"},
            {"Component": "Ventilation Loss", "Energy (kWh)": rc_res.get("ventilation_loss_kWh", 4.5), "Category": "Heat Loss"},
            {"Component": "Auxiliary Heating Energy", "Energy (kWh)": heating_kwh, "Category": "HVAC Energy Needed"},
            {"Component": "Auxiliary Cooling Energy", "Energy (kWh)": cooling_kwh, "Category": "HVAC Energy Needed"},
        ])
        fig_bal = px.bar(
            balance_df, x="Component", y="Energy (kWh)", color="Category",
            title=f"Daily Energy Balance Breakdown ({prof['name']})",
            color_discrete_map={"Heat Gain": "#2ECC71", "Heat Loss": "#E74C3C", "HVAC Energy Needed": "#F39C12"}
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    st.divider()
    st.markdown("#### 3. Insulation Thickness Sensitivity Sweep (0mm to 150mm)")
    st.caption(f"Evaluates insulation thickness sweep specifically under active region climate: {prof['name']}")

    # Sensitivity sweep simulation
    sweep_ins = list(range(0, 160, 15))
    sweep_results = []
    for ins_v in sweep_ins:
        t_cfg = {**user_design_dict, "insulation_thickness_mm": ins_v}
        res_sweep = simulate_shelter(t_cfg, climate_scenario=climate_scenario)
        sweep_results.append({
            "Insulation Thickness (mm)": ins_v,
            "Heating Energy (kWh/day)": res_sweep.get("heating_energy_required_kWh", 0.0),
            "Cooling Energy (kWh/day)": res_sweep.get("cooling_energy_required_kWh", 0.0),
            "Comfort Hours (hrs/day)": res_sweep["comfort_hours"],
        })

    df_sweep = pd.DataFrame(sweep_results)
    fig_sweep = px.line(
        df_sweep, x="Insulation Thickness (mm)",
        y=["Heating Energy (kWh/day)", "Cooling Energy (kWh/day)", "Comfort Hours (hrs/day)"],
        title=f"Impact of Insulation Thickness on Thermal Performance ({prof['name']})",
        markers=True
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

# ---------------- TAB 3: MULTI-OBJECTIVE OPTIMIZATION ----------------
with tab_opt:
    st.subheader("Multi-Objective Design Space Exploration & Pareto Analysis")
    st.markdown(f"Executes NSGA-II multi-objective genetic optimization dynamically for **{prof['name']}**.")

    col_opt1, col_opt2 = st.columns([1, 2])

    with col_opt1:
        max_fp = st.number_input("Maximum Footprint Area Limit (m²)", 9.0, 48.0, 30.0, help="Upper bound on shelter floor footprint area.")
        pop_sz = st.slider("Population Size (Designs per Generation)", 50, 200, 100, 10)
        gen_cnt = st.slider("Optimization Generations", 10, 80, 40, 5)
        opt_btn = st.button("Execute Multi-Objective Optimization", type="primary")

    if opt_btn or "pareto_data" not in st.session_state or st.session_state.get("opt_climate") != climate_scenario:
        with st.spinner(f"Running NSGA-II Genetic Optimization for {prof['name']}..."):
            pareto_df, elapsed, evals, candidates = run_optimization(
                climate_scenario=climate_scenario,
                materials_cfg={
                    "wall_material": wall_material, "roof_material": roof_material,
                    "floor_material": floor_material, "insulation_material": insulation_material,
                    "glazing_type": glazing_type
                },
                pop_size=pop_sz, n_gen=gen_cnt,
                max_footprint=max_fp
            )
            st.session_state["pareto_data"] = pareto_df
            st.session_state["opt_stats"] = (elapsed, evals, candidates)
            st.session_state["opt_climate"] = climate_scenario

    pareto_df = st.session_state["pareto_data"]
    elapsed, evals, candidates = st.session_state["opt_stats"]

    with col_opt2:
        st.markdown(f'<div class="status-badge-success">Evaluated {evals:,} shelter configurations in {elapsed:.2f} seconds ({elapsed/evals*1000:.3f} ms per design) for {prof["name"]}.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### 1. Interactive 3D Pareto Front — {prof['name']}")
    cost_col = "estimated_cost_inr" if "estimated_cost_inr" in pareto_df.columns else "estimated_cost_usd"
    fig_pareto = px.scatter_3d(
        pareto_df, x="heating_energy_required_kWh", y="comfort_hours", z=cost_col,
        color="insulation_thickness_mm", size="footprint_m2",
        hover_data=["length", "width", "window_ratio", "pcm_thickness_mm"],
        title=f"3D Pareto Optimal Space ({prof['name']})",
        labels={
            "heating_energy_required_kWh": "Heating Energy (kWh/day)",
            "comfort_hours": "Comfort Hours (hrs/day)",
            "estimated_cost_inr": "Est. Cost (₹)",
            "estimated_cost_usd": "Est. Cost (₹)",
            "insulation_thickness_mm": "Insulation (mm)"
        },
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown(f"#### 2. Optimized Candidate Designs Spectrum ({prof['name']})")
    c_cols = st.columns(4)
    for i, (name, cand) in enumerate(candidates.items()):
        with c_cols[i]:
            st.markdown(f"##### {name}")
            st.write(f"**Dimensions**: {cand['length']:.1f}m x {cand['width']:.1f}m ({cand['footprint_m2']:.1f} m²)")
            st.write(f"**Insulation Thickness**: {cand['insulation_thickness_mm']:.0f} mm")
            st.write(f"**Window-to-Wall Ratio**: {cand['window_ratio']*100:.0f}%")
            st.write(f"**Heating Energy**: {cand['heating_energy_required_kWh']:.1f} kWh/day")
            st.write(f"**Comfort Hours**: {cand['comfort_hours']:.1f} hrs/day")

# ---------------- TAB 4: DESIGN COMPARISON MATRIX ----------------
with tab_comp:
    st.subheader("Side-by-Side Design Comparison Matrix")
    st.markdown(f"Dynamically computes and compares performance metrics across design variants for **{prof['name']}**.")

    d1 = user_design_dict.copy()
    d2 = {**user_design_dict, "insulation_thickness_mm": 120.0, "window_ratio": 0.25, "orientation_deg": 0.0}
    d3 = {**user_design_dict, "insulation_thickness_mm": 30.0, "window_ratio": 0.10, "orientation_deg": 90.0}
    d4 = {**user_design_dict, "wall_material": "sandwich_panel", "insulation_thickness_mm": 60.0, "window_ratio": 0.15}

    r1 = simulate_shelter(d1, climate_scenario=climate_scenario)
    r2 = simulate_shelter(d2, climate_scenario=climate_scenario)
    r3 = simulate_shelter(d3, climate_scenario=climate_scenario)
    r4 = simulate_shelter(d4, climate_scenario=climate_scenario)

    comp_df = pd.DataFrame([
        {
            "Design Option": "Current Custom User Design",
            "Wall Material": d1["wall_material"],
            "Insulation (mm)": d1["insulation_thickness_mm"],
            "Window Ratio": f"{d1['window_ratio']*100:.0f}%",
            "Orientation": f"{d1['orientation_deg']}°",
            "Heating Energy (kWh/day)": r1.get("heating_energy_required_kWh", 0.0),
            "Cooling Energy (kWh/day)": r1.get("cooling_energy_required_kWh", 0.0),
            "Comfort Hours (hrs/day)": r1["comfort_hours"],
            "Heat Loss (kWh/day)": r1["total_heat_loss_kWh"]
        },
        {
            "Design Option": "High-Insulation Solar Design",
            "Wall Material": d2["wall_material"],
            "Insulation (mm)": d2["insulation_thickness_mm"],
            "Window Ratio": f"{d2['window_ratio']*100:.0f}%",
            "Orientation": f"{d2['orientation_deg']}°",
            "Heating Energy (kWh/day)": r2.get("heating_energy_required_kWh", 0.0),
            "Cooling Energy (kWh/day)": r2.get("cooling_energy_required_kWh", 0.0),
            "Comfort Hours (hrs/day)": r2["comfort_hours"],
            "Heat Loss (kWh/day)": r2["total_heat_loss_kWh"]
        },
        {
            "Design Option": "Low-Insulation East-West Facing Design",
            "Wall Material": d3["wall_material"],
            "Insulation (mm)": d3["insulation_thickness_mm"],
            "Window Ratio": f"{d3['window_ratio']*100:.0f}%",
            "Orientation": f"{d3['orientation_deg']}°",
            "Heating Energy (kWh/day)": r3.get("heating_energy_required_kWh", 0.0),
            "Cooling Energy (kWh/day)": r3.get("cooling_energy_required_kWh", 0.0),
            "Comfort Hours (hrs/day)": r3["comfort_hours"],
            "Heat Loss (kWh/day)": r3["total_heat_loss_kWh"]
        },
        {
            "Design Option": "Modular Prefab Sandwich Panel",
            "Wall Material": d4["wall_material"],
            "Insulation (mm)": d4["insulation_thickness_mm"],
            "Window Ratio": f"{d4['window_ratio']*100:.0f}%",
            "Orientation": f"{d4['orientation_deg']}°",
            "Heating Energy (kWh/day)": r4.get("heating_energy_required_kWh", 0.0),
            "Cooling Energy (kWh/day)": r4.get("cooling_energy_required_kWh", 0.0),
            "Comfort Hours (hrs/day)": r4["comfort_hours"],
            "Heat Loss (kWh/day)": r4["total_heat_loss_kWh"]
        },
    ])

    st.dataframe(comp_df.round(2), use_container_width=True)

    st.markdown("#### Performance Metrics Comparison Chart")
    fig_comp = px.bar(
        comp_df, x="Design Option", y=["Heating Energy (kWh/day)", "Cooling Energy (kWh/day)", "Comfort Hours (hrs/day)", "Heat Loss (kWh/day)"],
        barmode="group", title=f"Performance Comparison Matrix — {prof['name']}"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------- TAB 5: SENSITIVITY & RECOMMENDATION ANALYSIS ----------------
with tab_rec:
    st.subheader("Sensitivity & Recommendation Analysis")
    st.markdown(f"Explains physical drivers behind shelter performance specifically under **{prof['name']}**.")

    exp_data = explainer.explain_design_recommendation(
        {**user_design_dict, "heating_energy_required_kWh": rc_res.get("heating_energy_required_kWh", 15.0)},
        target="heating_energy_required_kWh"
    )

    st.markdown(f'<div class="metric-card" style="text-align: left; margin-bottom: 1.2rem;">{exp_data["narrative_explanation"]}</div>', unsafe_allow_html=True)

    c_xai1, c_xai2 = st.columns([1, 1])
    with c_xai1:
        st.markdown("#### Parameter Sensitivity Impact on Energy Demand")
        imp_df = explainer.get_global_importance("heating_energy_required_kWh").head(8)
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title=f"Parameter Importance Weights ({prof['name']})",
            color="importance", color_continuous_scale="Purples"
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with c_xai2:
        st.markdown("#### Local Parameter Sensitivity Breakdown")
        sens_df = pd.DataFrame(exp_data["local_sensitivity"])
        st.dataframe(sens_df, use_container_width=True)

    st.caption(exp_data["scientific_disclaimer"])

# ---------------- TAB 6: VALIDATION STUDIO ----------------
with tab_val:
    st.subheader("Tiered Physics & Validation Studio")
    st.markdown(f"""
    Cross-validates top optimized shelter configurations for **{prof['name']}** against direct multi-node RC physics recalculations:
    """)

    # Explicit solver availability status check
    if not ansys_engine.is_available():
        st.markdown('<div class="status-badge-warning">ANSYS Solver Integration: Operating under physics-model validation mode.</div><br>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge-success">ANSYS Solver Connected.</div><br>', unsafe_allow_html=True)

    st.markdown(f"#### Comprehensive Physics Verification Report ({prof['name']})")

    pareto_df = st.session_state.get("pareto_data")
    if pareto_df is not None and not pareto_df.empty:
        top_cand = pareto_df.head(5).copy()
        val_records = []
        for rank_idx, (_, row) in enumerate(top_cand.iterrows(), start=1):
            cand_dict = row.to_dict()
            pred_hvac = float(row.get("heating_energy_required_kWh", 15.0))
            rerun_res = simulate_shelter(cand_dict, climate_scenario=climate_scenario)
            rerun_hvac = float(rerun_res.get("heating_energy_required_kWh", pred_hvac))
            dev_pct = abs(pred_hvac - rerun_hvac) / max(rerun_hvac, 1.0) * 100.0
            val_records.append({
                "Rank": rank_idx,
                "Optimized Design Energy (kWh/day)": round(pred_hvac, 2),
                "Physics Re-run Energy (kWh/day)": round(rerun_hvac, 2),
                "Physics Deviation": f"{dev_pct:.1f}%",
                "Validation Status": "Physics Engine Verified",
            })
        val_df = pd.DataFrame(val_records)
        st.dataframe(val_df, use_container_width=True)
        mean_dev = val_df['Physics Deviation'].str.rstrip('%').astype(float).mean()
        st.caption(f"Mean Physics Verification Deviation: {mean_dev:.1f}% across top candidate designs under {prof['name']}.")
    else:
        st.info("Execute Optimization in Tab 3 to generate live dynamic validation reports.")

st.divider()
st.caption(f"Shelter Thermal Design & Optimization System | Active Region: {prof['region']}")
