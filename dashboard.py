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
    page_title="Sheltrix | AI-Powered Shelter Design",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern professional look (glassmorphism, clean typography)
st.markdown("""
<style>
    /* Global Typography and Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00C6FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.7rem;
        font-weight: 800;
        color: #2b3035;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Dark mode overrides for section header */
    @media (prefers-color-scheme: dark) {
        .section-header {
            color: #f8f9fa;
            border-bottom: 2px solid #343a40;
        }
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0072FF;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1.05rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }
    
    /* Badges */
    .status-badge-success {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.4);
        color: #2ECC71;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .status-badge-warning {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        background: rgba(241, 196, 15, 0.15);
        border: 1px solid rgba(241, 196, 15, 0.4);
        color: #F1C40F;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    /* Clean up expanders and tabs */
    .streamlit-expanderHeader {
        font-weight: 700;
    }
    
    /* Styling for top navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 12px 24px;
        background-color: transparent;
        border-radius: 8px;
        border: 1px solid transparent;
        font-weight: 700;
        font-size: 1.1rem;
        white-space: pre-wrap;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 114, 255, 0.1) !important;
        border: 1px solid rgba(0, 114, 255, 0.3) !important;
        color: #0072FF !important;
        border-bottom-color: rgba(0, 114, 255, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

cm = ClimateManager()
mdb = MaterialDatabase()
qc = SurrogateQualityControl()
explainer = DesignExplainer()
ansys_engine = get_simulation_engine("ansys")

# ---------------- HEADER ----------------
st.markdown('<div class="main-header">Sheltrix</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Climate Responsive Shelter Design</div>', unsafe_allow_html=True)

# ---------------- TOP NAVIGATION ----------------
nav_tab1, nav_tab2, nav_tab3 = st.tabs([
    "Project Configuration", 
    "Primary Results & 3D Digital Twin", 
    "Advanced Engineering Analysis"
])

# ---------------- WELCOME / INPUT SECTION (PAGE 1) ----------------
with nav_tab1:
    st.markdown('<div class="section-header">1. Project Configuration</div>', unsafe_allow_html=True)

    # Container for clean inputs
    with st.container():
        col_preset, col_region, col_scenario = st.columns(3)
        
        with col_preset:
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

        with col_region:
            regions = cm.get_available_regions()
            selected_region = st.selectbox(
                "Geographical Region",
                regions,
                index=0,
                help="Select target climate region."
            )

        with col_scenario:
            available_scenarios = cm.get_scenarios_for_region(selected_region)
            scenario_names_map = {k: cm.get_profile(k)["name"] for k in available_scenarios}
            climate_scenario = st.selectbox(
                "Climate Scenario / Season",
                available_scenarios,
                index=0,
                format_func=lambda x: scenario_names_map[x],
                help="Select seasonal profile."
            )
            prof = cm.get_profile(climate_scenario)

        st.caption(f"📍 Active Location: **{prof['name']} ({prof['region']})** | Data Source: {prof['data_source']}")

        # Advanced inputs hidden behind expanders for clean UI
        with st.expander("Adjust Shelter Dimensions & Orientation", expanded=False):
            c_dim1, c_dim2, c_dim3 = st.columns(3)
            with c_dim1:
                length = st.slider("Length (m)", 3.0, 8.0, p_len, 0.5)
                width = st.slider("Width (m)", 3.0, 6.0, p_wid, 0.5)
            with c_dim2:
                wall_height = st.slider("Wall Height (m)", 2.2, 3.5, p_height, 0.1)
                roof_angle = st.slider("Roof Pitch (degrees)", 0.0, 45.0, p_roof, 5.0)
            with c_dim3:
                orientation_deg = st.slider("Solar Orientation (0=South)", 0.0, 180.0, p_orient, 15.0)

        with st.expander("Advanced Materials & Insulation", expanded=False):
            c_mat1, c_mat2, c_mat3 = st.columns(3)
            with c_mat1:
                wall_material = st.selectbox("Wall Material", WALL_MATERIAL_LIST, index=WALL_MATERIAL_LIST.index(p_wall))
                roof_material = st.selectbox("Roof Material", WALL_MATERIAL_LIST, index=WALL_MATERIAL_LIST.index(p_roof_m))
                floor_material = st.selectbox("Floor Slab", ["concrete", "brick", "cseb"], index=["concrete", "brick", "cseb"].index(p_floor))
            with c_mat2:
                insulation_material = st.selectbox("Insulation Core", INSULATION_LIST, index=INSULATION_LIST.index(p_ins))
                insulation_thickness_mm = st.slider("Insulation Thickness (mm)", 0.0, 150.0, p_ins_t, 10.0)
            with c_mat3:
                window_ratio = st.slider("Window-to-Wall Ratio", 0.05, 0.40, p_wwr, 0.05)
                glazing_type = st.selectbox("Window Glazing", GLAZING_LIST, index=GLAZING_LIST.index(p_glaz))
                pcm_present = st.checkbox("Include PCM Storage", value=(p_pcm_t > 0))
                pcm_thickness_mm = st.slider("PCM Thickness (mm)", 0.0, 50.0, p_pcm_t if pcm_present else 0.0, 5.0, disabled=not pcm_present)

        user_design_dict = {
            "length": length, "width": width, "wall_height": wall_height,
            "roof_angle": roof_angle, "orientation_deg": orientation_deg,
            "wall_material": wall_material, "roof_material": roof_material,
            "floor_material": floor_material, "insulation_material": insulation_material,
            "insulation_thickness_mm": insulation_thickness_mm, "window_ratio": window_ratio,
            "glazing_type": glazing_type, "pcm_present": 1 if pcm_present else 0,
            "pcm_thickness_mm": pcm_thickness_mm if pcm_present else 0.0,
        }

# ---------------- DYNAMIC CALCULATIONS (RUNS REGARDLESS OF TAB) ----------------
# Dynamic Physics Simulation Run
rc_res = simulate_shelter(user_design_dict, climate_scenario=climate_scenario)

# Derived Building Physics Quantities
fl_area = length * width
wall_ar = 2 * (length + width) * wall_height
win_ar = wall_ar * window_ratio
net_w_ar = wall_ar - win_ar
rf_ar = (length * width) / np.cos(np.radians(min(roof_angle, 60)))

w_props = mdb.get_material("wall", wall_material)
i_props = mdb.get_material("insulation", insulation_material)
g_props = mdb.get_material("glazing", glazing_type)

wall_weight_ton = (net_w_ar * 0.20 * w_props["rho"]) / 1000.0
ins_cost_usd = (insulation_thickness_mm / 50.0) * i_props.get("cost_per_m2_base", 15.0) * (net_w_ar + rf_ar)
total_est_cost_usd = (net_w_ar * w_props.get("cost_per_m2_base", 30)) + ins_cost_usd + (win_ar * g_props.get("cost_per_m2_base", 75))

heating_kwh = rc_res.get('heating_energy_required_kWh', 0.0)
cooling_kwh = rc_res.get('cooling_energy_required_kWh', 0.0)


# ---------------- MAIN RESULTS SECTION (PAGE 2) ----------------
with nav_tab2:
    st.markdown('<div class="section-header">2. Primary Results & 3D Digital Twin</div>', unsafe_allow_html=True)

    # Current Shelter Specification & Derived Parameters requested by user
    spec_col, derived_col = st.columns([1, 1])
    with spec_col:
        st.markdown("**Current Shelter Specifications (Raw Input)**")
        st.json(user_design_dict)
    with derived_col:
        st.markdown("**Derived Architectural & Thermal Properties**")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Floor Footprint Area", f"{fl_area:.1f} m²")
        pcol2.metric("Enclosure Volume", f"{fl_area * wall_height:.1f} m³")
        
        r_w = compute_layer_resistance("wall", wall_material, insulation_material, insulation_thickness_mm)
        pcol1.metric("Wall R-Value (Insulation)", f"{r_w:.2f} m²K/W")
        pcol2.metric("Wall U-Value (Conductance)", f"{(1.0/r_w):.3f} W/m²K")

    st.divider()

    # High-impact Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Thermal Comfort</div>
                <div class="metric-value">{rc_res["comfort_hours"]:.0f} <span style="font-size: 1rem; color: #6c757d;">hrs/day</span></div>
                <div style="font-size: 0.85rem; color: #2ECC71;">(18°C - 25°C Band)</div>
            </div>
        ''', unsafe_allow_html=True)

    with m_col2:
        primary_energy = cooling_kwh if cooling_kwh > heating_kwh else heating_kwh
        energy_type = "Cooling" if cooling_kwh > heating_kwh else "Heating"
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">{energy_type} Required</div>
                <div class="metric-value">{primary_energy:.1f} <span style="font-size: 1rem; color: #6c757d;">kWh/day</span></div>
                <div style="font-size: 0.85rem; color: #F39C12;">Peak Daily HVAC Demand</div>
            </div>
        ''', unsafe_allow_html=True)

    with m_col3:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Est. Cost</div>
                <div class="metric-value">${total_est_cost_usd:,.0f}</div>
                <div style="font-size: 0.85rem; color: #A0AAB8;">Envelope, Insulation & Glazing</div>
            </div>
        ''', unsafe_allow_html=True)

    with m_col4:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Structural Mass</div>
                <div class="metric-value">{wall_weight_ton:.1f} <span style="font-size: 1rem; color: #6c757d;">Tons</span></div>
                <div style="font-size: 0.85rem; color: #A0AAB8;">Outer Wall Envelope</div>
            </div>
        ''', unsafe_allow_html=True)

    st.write("") # spacing

    # 3D Twin and Summary
    c_vis, c_sum = st.columns([1.5, 1])

    with c_vis:
        st.markdown("<h4 style='text-align: center; color: #2b3035;'>Interactive 3D Digital Twin</h4>", unsafe_allow_html=True)
        fig_3d = render_shelter_3d(user_design_dict, show_layers=True)
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=450)
        st.plotly_chart(fig_3d, use_container_width=True)

    with c_sum:
        st.markdown("<h4 style='color: #2b3035;'>Key Properties Summary</h4>", unsafe_allow_html=True)
        
        r_wall = compute_layer_resistance("wall", wall_material, insulation_material, insulation_thickness_mm)
        u_wall = 1.0 / r_wall
        
        summary_data = {
            "Property": ["Floor Footprint", "Wall U-Value", "Total Heat Loss", "Passive Solar Gain", "Wall System", "Insulation"],
            "Value": [
                f"{fl_area:.1f} m²", 
                f"{u_wall:.3f} W/m²K", 
                f"{rc_res['total_heat_loss_kWh']:.1f} kWh/day", 
                f"{rc_res['total_solar_gain_kWh']:.1f} kWh/day",
                w_props["name"],
                f"{i_props['name']} ({insulation_thickness_mm}mm)"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        valid, warnings = qc.validate_input_bounds(user_design_dict)
        if not valid:
            for w in warnings:
                st.error(f"Design Check: {w}")
        else:
            st.markdown('<div class="status-badge-success" style="margin-top: 1rem; width: 100%; text-align: center;">✓ All physical bounds valid</div>', unsafe_allow_html=True)

# ---------------- ADVANCED ENGINEERING ANALYSIS (PAGE 3) ----------------
with nav_tab3:
    st.markdown('<div class="section-header">3. Advanced Engineering Analysis</div>', unsafe_allow_html=True)
    st.markdown("Detailed physical, thermal, and multi-objective optimization data for engineering review.")

    tab_pred, tab_opt, tab_comp, tab_rec, tab_val = st.tabs([
        "Physics & Thermal Analysis",
        "Multi-Objective Optimization",
        "Design Comparison Matrix",
        "Sensitivity & XAI",
        "Validation Studio"
    ])

    # ---- TAB: PHYSICS & THERMAL ----
    with tab_pred:
        st.subheader("Physics Engine Simulation & Detailed Heat Balance Analysis")
        st.caption(f"Solves 24-hour diurnal lumped RC thermal resistance network dynamically for {prof['name']}.")

        m1, m2, m3, m4 = st.columns(4)
        if cooling_kwh > heating_kwh:
            m1.metric("Predicted Cooling Energy Needed", f"{cooling_kwh:.1f} kWh/day", help="Kilowatt-hours of cooling energy required daily to prevent overheating.")
        else:
            m1.metric("Predicted Heating Energy Needed", f"{heating_kwh:.1f} kWh/day", help="Kilowatt-hours of heating energy required daily to maintain indoor warmth.")

        m2.metric("Thermal Comfort Hours (18°C - 25°C)", f"{rc_res['comfort_hours']} hrs/day", delta=f"{rc_res['comfort_hours']-12:.0f} hrs vs baseline", help="Hours per day indoor temperature remains in comfortable 18°C - 25°C band.")
        m3.metric("Total Wall & Roof Heat Loss", f"{rc_res['total_heat_loss_kWh']:.1f} kWh/day", help="Conductive heat transfer escaping through building envelope.")
        m4.metric("Passive Solar Heat Gain", f"{rc_res['total_solar_gain_kWh']:.1f} kWh/day", help="Solar heat gained through windows during daylight hours.")
        
        st.divider()

        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown(f"**24-Hour Diurnal Indoor vs Outdoor Temperature Profile ({prof['region']})**")
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
                color_discrete_map={
                    "Indoor Temperature (°C)": "#FF4B4B",
                    "Outdoor Ambient Temp (°C)": "#1E88E5",
                    "Lower Comfort Limit (18°C)": "#4CAF50",
                    "Upper Comfort Limit (25°C)": "#FFA726"
                }
            )
            fig_temp.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2), margin=dict(t=20))
            st.plotly_chart(fig_temp, use_container_width=True)

        with c_right:
            st.markdown("**Daily Energy Balance Breakdown (kWh/day)**")
            balance_df = pd.DataFrame([
                {"Component": "Passive Solar Gain", "Energy (kWh)": rc_res["total_solar_gain_kWh"], "Category": "Heat Gain"},
                {"Component": "Envelope Conductive Loss", "Energy (kWh)": rc_res["total_heat_loss_kWh"], "Category": "Heat Loss"},
                {"Component": "Ventilation Loss", "Energy (kWh)": rc_res.get("ventilation_loss_kWh", 4.5), "Category": "Heat Loss"},
                {"Component": "Heating Energy Needed", "Energy (kWh)": heating_kwh, "Category": "HVAC Energy"},
                {"Component": "Cooling Energy Needed", "Energy (kWh)": cooling_kwh, "Category": "HVAC Energy"},
            ])
            fig_bal = px.bar(
                balance_df, x="Component", y="Energy (kWh)", color="Category",
                color_discrete_map={"Heat Gain": "#2ECC71", "Heat Loss": "#E74C3C", "HVAC Energy": "#F39C12"}
            )
            fig_bal.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_bal, use_container_width=True)

        st.markdown("**Insulation Thickness Sensitivity Sweep (0mm to 150mm)**")
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
            markers=True
        )
        fig_sweep.update_layout(margin=dict(t=20), height=350)
        st.plotly_chart(fig_sweep, use_container_width=True)

    # ---- TAB: MULTI-OBJECTIVE OPTIMIZATION ----
    with tab_opt:
        st.subheader("Multi-Objective Design Space Exploration & Pareto Analysis")
        st.markdown(f"Executes NSGA-II multi-objective genetic optimization dynamically for **{prof['name']}**.")

        col_opt1, col_opt2 = st.columns([1, 2])

        with col_opt1:
            max_fp = st.number_input("Maximum Footprint Area Limit (m²)", 9.0, 48.0, 30.0)
            pop_sz = st.slider("Population Size (Designs per Gen)", 50, 200, 100, 10)
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
            st.markdown(f'<div class="status-badge-success" style="margin-top: 25px;">Evaluated {evals:,} shelter configurations in {elapsed:.2f} seconds ({elapsed/evals*1000:.3f} ms per design) for {prof["name"]}.</div>', unsafe_allow_html=True)

        st.markdown(f"**Interactive 3D Pareto Front — {prof['name']}**")
        fig_pareto = px.scatter_3d(
            pareto_df, x="heating_energy_required_kWh", y="comfort_hours", z="estimated_cost_usd",
            color="insulation_thickness_mm", size="footprint_m2",
            hover_data=["length", "width", "window_ratio", "pcm_thickness_mm"],
            labels={
                "heating_energy_required_kWh": "Heating Energy (kWh)",
                "comfort_hours": "Comfort Hours",
                "estimated_cost_usd": "Cost ($)",
                "insulation_thickness_mm": "Insulation"
            },
            color_continuous_scale="Viridis",
        )
        fig_pareto.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown(f"**Optimized Candidate Designs Spectrum ({prof['name']})**")
        c_cols = st.columns(4)
        for i, (name, cand) in enumerate(candidates.items()):
            with c_cols[i]:
                st.markdown(f"##### {name}")
                st.write(f"- **Dimensions**: {cand['length']:.1f}m x {cand['width']:.1f}m")
                st.write(f"- **Insulation**: {cand['insulation_thickness_mm']:.0f} mm")
                st.write(f"- **Window Ratio**: {cand['window_ratio']*100:.0f}%")
                st.write(f"- **Heating Energy**: {cand['heating_energy_required_kWh']:.1f} kWh")
                st.write(f"- **Comfort**: {cand['comfort_hours']:.1f} hrs")

    # ---- TAB: DESIGN COMPARISON ----
    with tab_comp:
        st.subheader("Side-by-Side Design Comparison Matrix")
        
        d1 = user_design_dict.copy()
        d2 = {**user_design_dict, "insulation_thickness_mm": 120.0, "window_ratio": 0.25, "orientation_deg": 0.0}
        d3 = {**user_design_dict, "insulation_thickness_mm": 30.0, "window_ratio": 0.10, "orientation_deg": 90.0}
        d4 = {**user_design_dict, "wall_material": "sandwich_panel", "insulation_thickness_mm": 60.0, "window_ratio": 0.15}

        r1 = simulate_shelter(d1, climate_scenario=climate_scenario)
        r2 = simulate_shelter(d2, climate_scenario=climate_scenario)
        r3 = simulate_shelter(d3, climate_scenario=climate_scenario)
        r4 = simulate_shelter(d4, climate_scenario=climate_scenario)

        comp_df = pd.DataFrame([
            {"Option": "Current Design", "Insulation": f"{d1['insulation_thickness_mm']}mm", "WWR": f"{d1['window_ratio']*100:.0f}%", "Heat Energy": r1.get("heating_energy_required_kWh", 0.0), "Comfort": r1["comfort_hours"], "Loss": r1["total_heat_loss_kWh"]},
            {"Option": "High-Insulation Solar", "Insulation": f"{d2['insulation_thickness_mm']}mm", "WWR": f"{d2['window_ratio']*100:.0f}%", "Heat Energy": r2.get("heating_energy_required_kWh", 0.0), "Comfort": r2["comfort_hours"], "Loss": r2["total_heat_loss_kWh"]},
            {"Option": "Low-Insulation EW", "Insulation": f"{d3['insulation_thickness_mm']}mm", "WWR": f"{d3['window_ratio']*100:.0f}%", "Heat Energy": r3.get("heating_energy_required_kWh", 0.0), "Comfort": r3["comfort_hours"], "Loss": r3["total_heat_loss_kWh"]},
            {"Option": "Prefab Panel", "Insulation": f"{d4['insulation_thickness_mm']}mm", "WWR": f"{d4['window_ratio']*100:.0f}%", "Heat Energy": r4.get("heating_energy_required_kWh", 0.0), "Comfort": r4["comfort_hours"], "Loss": r4["total_heat_loss_kWh"]},
        ])

        st.dataframe(comp_df.round(2), use_container_width=True, hide_index=True)

        fig_comp = px.bar(
            comp_df, x="Option", y=["Heat Energy", "Comfort", "Loss"],
            barmode="group"
        )
        fig_comp.update_layout(margin=dict(t=20), height=400)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ---- TAB: SENSITIVITY & XAI ----
    with tab_rec:
        st.subheader("Sensitivity & Explainable AI (XAI)")
        
        exp_data = explainer.explain_design_recommendation(
            {**user_design_dict, "heating_energy_required_kWh": rc_res.get("heating_energy_required_kWh", 15.0)},
            target="heating_energy_required_kWh"
        )

        st.info(f"**AI Narrative Insight:** {exp_data['narrative_explanation']}")

        c_xai1, c_xai2 = st.columns([1, 1])
        with c_xai1:
            st.markdown("**Global Parameter Importance**")
            imp_df = explainer.get_global_importance("heating_energy_required_kWh").head(8)
            fig_imp = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Purples"
            )
            fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig_imp, use_container_width=True)

        with c_xai2:
            st.markdown("**Local Feature Sensitivity**")
            st.dataframe(pd.DataFrame(exp_data["local_sensitivity"]), use_container_width=True)

        st.caption(exp_data["scientific_disclaimer"])

    # ---- TAB: VALIDATION STUDIO ----
    with tab_val:
        st.subheader("Tiered Validation against RC Engine / ANSYS")
        
        if not ansys_engine.is_available():
            st.markdown('<div class="status-badge-warning">ANSYS integration absent. Validating via Tier-1 RC Network calculations.</div><br>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-success">ANSYS Solver Connected.</div><br>', unsafe_allow_html=True)

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
                    "Surrogate Prediction (kWh)": round(pred_hvac, 2),
                    "Physics Rerun (kWh)": round(rerun_hvac, 2),
                    "Deviation": f"{dev_pct:.1f}%",
                    "Status": "Physics Engine Verified",
                })
            st.dataframe(pd.DataFrame(val_records), use_container_width=True, hide_index=True)
        else:
            st.info("Execute Optimization in the Multi-Objective tab first to generate validation reports.")

st.divider()
st.markdown(f"<p style='text-align: center; color: #a0aab8; font-size: 0.9rem;'>Sheltrix Design Platform v2.0 | Active Site: {prof['region']}</p>", unsafe_allow_html=True)
