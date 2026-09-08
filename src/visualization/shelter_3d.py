"""
Module 13: 3D Shelter Interactive Visualization
Renders 3D parametric shelter geometry using Plotly Mesh3d/Scatter3d.
Visualizes dimensions, roof angle pitch, window glazing cutouts, solar orientation vector,
and insulation/PCM layer toggles.
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any


def render_shelter_3d(params: Dict[str, Any], show_layers: bool = True) -> go.Figure:
    """
    Generate interactive Plotly 3D mesh representation of shelter geometry.
    """
    L = float(params.get("length", 5.0))
    W = float(params.get("width", 4.0))
    H = float(params.get("wall_height", 2.8))
    roof_angle = float(params.get("roof_angle", 20.0))
    orient_deg = float(params.get("orientation_deg", 0.0))
    window_ratio = float(params.get("window_ratio", 0.15))
    ins_t = float(params.get("insulation_thickness_mm", 80.0)) / 1000.0

    # Main wall box 8 vertices (x, y, z) centered around origin (0,0,0)
    x_half, y_half = L / 2.0, W / 2.0

    vertices = np.array([
        [-x_half, -y_half, 0], # 0: bottom-left-front
        [ x_half, -y_half, 0], # 1: bottom-right-front
        [ x_half,  y_half, 0], # 2: bottom-right-back
        [-x_half,  y_half, 0], # 3: bottom-left-back
        [-x_half, -y_half, H], # 4: top-left-front
        [ x_half, -y_half, H], # 5: top-right-front
        [ x_half,  y_half, H], # 6: top-right-back
        [-x_half,  y_half, H], # 7: top-left-back
    ])

    # Roof peak ridge height (pitched along length)
    roof_peak_h = H + (y_half * np.tan(np.radians(min(roof_angle, 45.0))))
    roof_ridge_vertices = np.array([
        [-x_half, 0, roof_peak_h], # 8: ridge left
        [ x_half, 0, roof_peak_h], # 9: ridge right
    ])

    all_verts = np.vstack([vertices, roof_ridge_vertices])

    # Define wall & roof mesh triangles (i, j, k)
    triangles = [
        # Wall faces
        [0, 1, 5], [0, 5, 4], # Front wall
        [1, 2, 6], [1, 6, 5], # Right wall
        [2, 3, 7], [2, 7, 6], # Back wall
        [3, 0, 4], [3, 4, 7], # Left wall
        # Roof pitch faces
        [4, 5, 9], [4, 9, 8], # Front roof slope
        [6, 7, 8], [6, 8, 9], # Back roof slope
        [4, 8, 7],            # Left gable triangle
        [5, 6, 9],            # Right gable triangle
    ]
    triangles = np.array(triangles)

    fig = go.Figure()

    # 1. Structural Shell Mesh
    wall_mat_name = params.get("wall_material", "CSEB Earth Block").upper()
    fig.add_trace(go.Mesh3d(
        x=all_verts[:, 0], y=all_verts[:, 1], z=all_verts[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color="sandybrown" if "cseb" in wall_mat_name.lower() or "brick" in wall_mat_name.lower() else "lightslategray",
        opacity=0.85,
        name=f"Structure ({wall_mat_name})",
        hoverinfo="name",
    ))

    # 2. Window Glazing Cutout Visualizer (on South-facing front wall)
    win_w = L * np.sqrt(window_ratio) * 0.8
    win_h = H * np.sqrt(window_ratio) * 0.8
    win_z0 = H * 0.35

    win_verts = np.array([
        [-win_w/2, -y_half - 0.02, win_z0],
        [ win_w/2, -y_half - 0.02, win_z0],
        [ win_w/2, -y_half - 0.02, win_z0 + win_h],
        [-win_w/2, -y_half - 0.02, win_z0 + win_h],
    ])
    fig.add_trace(go.Mesh3d(
        x=win_verts[:, 0], y=win_verts[:, 1], z=win_verts[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="deepskyblue", opacity=0.9,
        name=f"Window Glazing ({params.get('glazing_type', 'double_pane')})",
    ))

    # 3. Solar Orientation Vector Arrow
    rad = np.radians(orient_deg)
    arrow_len = W * 0.8
    dx, dy = arrow_len * np.sin(rad), -arrow_len * np.cos(rad)
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[-y_half, -y_half - dy], z=[H/2, H/2],
        mode="lines+markers",
        line=dict(color="gold", width=6),
        marker=dict(symbol="diamond", size=6, color="gold"),
        name=f"Solar Exposure Vector ({orient_deg:.0f}° South)",
    ))

    # 4. Layer toggle (Insulation & PCM indicator bounds)
    if show_layers and ins_t > 0:
        # Insulation outer offset box
        ins_verts = vertices * (1.0 + ins_t / W)
        fig.add_trace(go.Scatter3d(
            x=ins_verts[[4,5,6,7,4], 0],
            y=ins_verts[[4,5,6,7,4], 1],
            z=ins_verts[[4,5,6,7,4], 2],
            mode="lines",
            line=dict(color="limegreen", width=4, dash="dash"),
            name=f"Insulation Layer ({ins_t*1000:.0f}mm {params.get('insulation_material', 'xps_foam')})",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Length (m)", range=[-L, L]),
            yaxis=dict(title="Width (m)", range=[-W, W]),
            zaxis=dict(title="Height (m)", range=[0, H + 2.0]),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D Shelter Digital Twin Model ({L:.1f}m x {W:.1f}m x {H:.1f}m)",
        legend=dict(x=0.01, y=0.99),
    )

    return fig


if __name__ == "__main__":
    test_params = {"length": 5.0, "width": 4.0, "wall_height": 2.8, "roof_angle": 20, "orientation_deg": 0, "window_ratio": 0.20, "insulation_thickness_mm": 80, "wall_material": "cseb"}
    fig = render_shelter_3d(test_params)
    print("3D Figure generated successfully.")
