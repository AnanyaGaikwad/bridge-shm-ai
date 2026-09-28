"""
Streamlit Web Dashboard for Bridge Structural Health Monitoring (SHM)
Deep Learning Anomaly Detection, Real-time Simulation, and Health Prediction.
"""

import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from src.data.dataset import load_sensor_csv
from src.data.preprocessing import (
    butter_bandpass_filter,
    compute_fft_spectrum,
    compute_spectrogram,
    create_sliding_windows,
)
from src.models.conv1d_bilstm_autoencoder import Conv1DBiLSTMAutoencoder
from src.models.health_index import HealthAssessment, StructuralHealthPredictor
from src.pipeline import BridgeSHMPipeline
from src.simulation.bridge_physics import BridgeDamageScenario, BridgeSimulator
from src.training.evaluator import ModelEvaluator

st.set_page_config(
    page_title="BridgeGuard AI — Bridge Structural Health Monitoring",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate Design System (High Contrast Dark Theme) ─────────────────────
st.markdown("""
<style>
/* ── Colour Tokens ─────────────────────────────────────────────── */
:root {
    --bg-main:        #0e131f;
    --bg-card:        #1E293B;
    --bg-sidebar:     #111827;
    --border-color:   #334155;
    --text-primary:   #FFFFFF;
    --text-secondary: #E2E8F0;
    --text-muted:     #94A3B8;
    --accent:         #2563eb;
    --accent-hover:   #1d4ed8;
    --font-sans:      "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}

/* ── Global Typography & Background ─────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-main) !important;
    color: var(--text-secondary) !important;
}

.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1280px !important;
    background-color: var(--bg-main) !important;
}

/* ── Titles, Headers & Subheaders ───────────────────────────────── */
h1, .stMarkdown h1, .main-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
    border-bottom: 2px solid var(--accent) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
    line-height: 1.25 !important;
}

.main-subtitle {
    font-size: 1rem !important;
    color: #94A3B8 !important;
    line-height: 1.6 !important;
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
}

h2, .stMarkdown h2, [data-testid="stHeader"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.6rem !important;
}

h3, h4, h5, h6, .stMarkdown h3, .stMarkdown h4 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
}

/* ── Body Text & Markdown ───────────────────────────────────────── */
p:not([style*="font-size"]), li, span:not([style*="font-size"]), .stMarkdown p:not([style*="font-size"]), .stMarkdown li, .stMarkdown span:not([style*="font-size"]) {
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: var(--text-secondary) !important;
}

strong, b, .stMarkdown strong {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Slider Controls & Labels (High Contrast White) ─────────────── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
[data-testid="stSlider"] [data-testid="stWidgetLabel"] span {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stSlider"] div[data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #CBD5E1 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid #FFFFFF !important;
}

/* ── Selectbox & Dropdowns ──────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: #FFFFFF !important;
    border-radius: 4px !important;
}

[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] li {
    background-color: var(--bg-card) !important;
    color: #FFFFFF !important;
    border-color: var(--border-color) !important;
}

/* ── Radio Buttons ──────────────────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stRadio"] [data-baseweb="radio"] label,
[data-testid="stRadio"] [data-baseweb="radio"] label span {
    color: var(--text-secondary) !important;
}

/* ── Sidebar Styling ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    border-bottom: 1px solid var(--border-color) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.75rem !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label {
    font-size: 0.84rem !important;
    padding: 0.35rem 0 !important;
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    color: #FFFFFF !important;
}

/* ── Metric Cards ───────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}

/* ── Control Panel Card Containers ──────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"] > div[style*="border"],
div:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1E293B !important;
    padding: 16px !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25) !important;
    margin-bottom: 0.8rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: transparent !important;
}

[data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    font-size: 0.95rem !important;
    border-bottom: 1px solid #334155 !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 0.8rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px !important;
    border-radius: 6px !important;
    width: 100% !important;
    border: none !important;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    border: none !important;
}

.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    background-color: #1E40AF !important;
    transform: translateY(1px) !important;
}

.stButton > button {
    background-color: var(--bg-card) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: #FFFFFF !important;
}

.stDownloadButton > button {
    background-color: var(--bg-card) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: #FFFFFF !important;
}

/* ── DataFrames & Tables ────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    background-color: var(--bg-card) !important;
}

/* ── Alerts & Info Banners ──────────────────────────────────────── */
[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}

[data-testid="stAlert"] p, [data-testid="stAlert"] div {
    color: #FFFFFF !important;
}

/* ── Expanders ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

/* ── File Uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] section {
    background-color: var(--bg-card) !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 4px !important;
}

[data-testid="stFileUploader"] section * {
    color: var(--text-secondary) !important;
}

/* ── Horizontal Divider ─────────────────────────────────────────── */
hr {
    border-color: var(--border-color) !important;
    margin: 1.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

CHECKPOINT_PATH = "checkpoints/bridge_shm_pipeline.pt"


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load or initialize pipeline."""
    pipeline = BridgeSHMPipeline(
        num_sensors=8,
        window_size=256,
        step_size=64,
        sampling_rate=100.0,
    )
    if os.path.exists(CHECKPOINT_PATH):
        pipeline.load_checkpoint(CHECKPOINT_PATH)
    else:
        # Quick baseline training if no checkpoint exists
        pipeline.train_on_baseline(duration_s=120.0, epochs=10, batch_size=32)
        pipeline.save_checkpoint(CHECKPOINT_PATH)
    return pipeline


def apply_chart_theme(fig: go.Figure, height: int = 300, title: str = None) -> go.Figure:
    """Apply neutral corporate dark styling to Plotly figures for high contrast."""
    update_dict = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color="#E2E8F0", family="Inter, Segoe UI, system-ui, sans-serif"),
    }
    if height:
        update_dict["height"] = height
    if title:
        update_dict["title"] = dict(text=title, font=dict(color="#FFFFFF", size=14))
    fig.update_layout(**update_dict)
    fig.update_xaxes(gridcolor="#1e293b", linecolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", linecolor="#334155", zerolinecolor="#334155")
    return fig


def render_bridge_schematic(highlight_sensor: str = "", damage_loc: float = None):
    """Render interactive 2D bridge schematic showing spans, piers, and sensors."""
    sim = BridgeSimulator()
    fig = go.Figure()

    # Bridge Deck (220m length)
    fig.add_trace(
        go.Scatter(
            x=[0, 220],
            y=[0, 0],
            mode="lines",
            line=dict(color="#3b82f6", width=8),
            name="Bridge Deck (220m)",
            hoverinfo="name",
        )
    )

    # Abutments & Piers
    # Left Abutment (0m)
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-8, 0],
            mode="lines",
            line=dict(color="#94a3b8", width=6),
            name="Abutment 1",
            showlegend=False,
        )
    )
    # Pier 1 (60m)
    fig.add_trace(
        go.Scatter(
            x=[60, 60],
            y=[-15, 0],
            mode="lines",
            line=dict(color="#cbd5e1", width=8),
            name="Pier 1 Support",
            showlegend=False,
        )
    )
    # Pier 2 (160m)
    fig.add_trace(
        go.Scatter(
            x=[160, 160],
            y=[-15, 0],
            mode="lines",
            line=dict(color="#cbd5e1", width=8),
            name="Pier 2 Support",
            showlegend=False,
        )
    )
    # Right Abutment (220m)
    fig.add_trace(
        go.Scatter(
            x=[220, 220],
            y=[-8, 0],
            mode="lines",
            line=dict(color="#94a3b8", width=6),
            name="Abutment 2",
            showlegend=False,
        )
    )

    # Span Boundary Annotations
    fig.add_vrect(x0=0, x1=60, fillcolor="rgba(59, 130, 246, 0.08)", line_width=0, annotation_text="Span 1 (60m)", annotation_position="top left", annotation_font=dict(color="#E2E8F0"))
    fig.add_vrect(x0=60, x1=160, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, annotation_text="Span 2 Main Span (100m)", annotation_position="top left", annotation_font=dict(color="#E2E8F0"))
    fig.add_vrect(x0=160, x1=220, fillcolor="rgba(59, 130, 246, 0.08)", line_width=0, annotation_text="Span 3 (60m)", annotation_position="top left", annotation_font=dict(color="#E2E8F0"))

    # Sensor Locations
    sensor_x = [s.location_m for s in sim.sensors]
    sensor_y = [0.0 for _ in sim.sensors]
    sensor_labels = [s.id for s in sim.sensors]
    sensor_descs = [f"{s.id}: {s.description} ({s.location_m}m)" for s in sim.sensors]

    colors = []
    sizes = []
    for s in sim.sensors:
        if s.id == highlight_sensor:
            colors.append("#ef4444")
            sizes.append(18)
        else:
            colors.append("#22c55e")
            sizes.append(13)

    fig.add_trace(
        go.Scatter(
            x=sensor_x,
            y=sensor_y,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, symbol="triangle-up", line=dict(width=1.5, color="#FFFFFF")),
            text=sensor_labels,
            textposition="top center",
            hovertext=sensor_descs,
            hoverinfo="text",
            name="Sensors (Accelerometers)",
        )
    )

    # Highlight damage injection location if active
    if damage_loc is not None:
        fig.add_trace(
            go.Scatter(
                x=[damage_loc],
                y=[0],
                mode="markers",
                marker=dict(size=22, color="rgba(239, 68, 68, 0.8)", symbol="x"),
                name="Damage Site",
                hoverinfo="name",
            )
        )

    fig.update_layout(
        title=dict(text="Bridge Digital Twin — Physical Span & Sensor Station Layout", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(title=dict(text="Bridge Longitudinal Position (meters)", font=dict(color="#E2E8F0")), range=[-10, 230], zeroline=False, gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(title="", showticklabels=False, range=[-18, 5], zeroline=False, gridcolor="#1e293b"),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
    )
    return fig


def main():
    st.markdown(
        """
        <div style="margin-top: -20px; margin-bottom: 24px;">
            <span style="font-size: 42px; font-weight: 800; color: #FFFFFF; display: block; line-height: 1.2; letter-spacing: -0.02em; margin-bottom: 8px;">BridgeGuard AI – Bridge Structural Health Monitoring</span>
            <span style="font-size: 16px; color: #94A3B8; display: block; line-height: 1.5;">Multi-Sensor Vibration Anomaly Detection, Dynamic Damage Localization, and Structural Health Index (SHI) Prediction powered by a Hybrid 1D-CNN + Bidirectional LSTM Autoencoder.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()

    # Sidebar Navigation & Settings
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Select Module:",
        [
            "Live Simulation & Damage Studio",
            "Anomaly Detection & Health Scoring",
            "Bridge Digital Twin & Frequencies",
            "Model Architecture & Performance",
            "Custom Sensor Data",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    st.sidebar.info(
        f"**Neural Engine:** PyTorch\n\n"
        f"**Compute Device:** `{pipeline.device}`\n\n"
        f"**Sensors Monitored:** 8 Distributed Accelerometer Channels\n\n"
        f"**Sampling Rate:** {pipeline.fs} Hz\n\n"
        f"**Window Size:** {pipeline.window_size} samples ({pipeline.window_size/pipeline.fs:.2f}s)\n\n"
        f"**Model Status:** {'Calibrated & Ready' if pipeline.is_trained else 'Untrained'}"
    )

    # Sensitivity Control
    sensitivity = st.sidebar.slider(
        "Threshold Sensitivity:",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher sensitivity makes the anomaly threshold tighter.",
    )
    pipeline.predictor.sensitivity = sensitivity
    if pipeline.predictor.is_calibrated:
        # Use the threshold calibrated and saved with the trained checkpoint.
        # Sensitivity only adjusts that calibrated baseline; it must not replace
        # the calibration with an unrelated mean + 2.5*std heuristic.
        if not hasattr(pipeline.predictor, "_calibrated_base_threshold"):
            pipeline.predictor._calibrated_base_threshold = float(pipeline.predictor.threshold)
        pipeline.predictor.threshold = pipeline.predictor._calibrated_base_threshold / sensitivity

    # =========================================================================
    # MODULE 1: LIVE SIMULATION & DAMAGE STUDIO
    # =========================================================================
    if menu == "Live Simulation & Damage Studio":
        st.header("Live Vibration Simulation & Structural Damage Studio")
        st.markdown(
            "Simulate dynamic bridge vibration under moving vehicular traffic, thermal environmental shifts, "
            "and inject structural faults (stiffness loss, cracks, bearing seizure) in real time."
        )

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            with st.container(border=True):
                st.subheader("Operational Conditions")
                sim_duration = st.slider("Duration (seconds):", 20, 90, 45, step=5)
                traffic_density = st.select_slider(
                    "Traffic Intensity:",
                    options=[0.5, 0.8, 1.0, 1.5, 2.0],
                    value=1.0,
                    format_func=lambda x: {0.5: "Light (0.5x)", 0.8: "Moderate (0.8x)", 1.0: "Standard (1.0x)", 1.5: "Heavy Highway (1.5x)", 2.0: "Rush Hour (2.0x)"}[x],
                )
                traffic_speed = st.slider("Mean Traffic Speed (km/h):", 40, 100, 65, step=5)

        with col_cfg2:
            with st.container(border=True):
                st.subheader("Environmental Factors")
                ambient_temp = st.slider("Ambient Temperature (°C):", 5.0, 45.0, 22.0, step=1.0, help="Temperature variation shifts Young's modulus E and baseline stiffness.")
                wind_level = st.slider("Wind Turbulence (g):", 0.01, 0.08, 0.03, step=0.01)
                sensor_noise = st.slider("Sensor Electronic Noise (mg):", 1.0, 10.0, 4.0, step=0.5)

        with col_cfg3:
            with st.container(border=True):
                st.subheader("Structural Damage Injection")
                damage_type = st.selectbox(
                    "Damage Scenario:",
                    ["None (Pristine Bridge)", "Stiffness Loss (Concrete Cracking / Corrosion)", "Impact Breathing Crack", "Bearing Seizure (Frozen Support)"],
                )
                dmg_type_code = {
                    "None (Pristine Bridge)": "none",
                    "Stiffness Loss (Concrete Cracking / Corrosion)": "stiffness_loss",
                    "Impact Breathing Crack": "crack",
                    "Bearing Seizure (Frozen Support)": "bearing_stiffness",
                }[damage_type]

                if dmg_type_code != "none":
                    damage_loc = st.slider("Damage Location along Deck (meters):", 10.0, 210.0, 110.0, step=5.0)
                    damage_sev = st.slider("Damage Severity (%):", 5, 50, 25, step=5) / 100.0
                    damage_start = st.slider("Damage Onset Time (s):", 5.0, float(sim_duration - 5), 15.0, step=1.0)
                else:
                    damage_loc = 110.0
                    damage_sev = 0.0
                    damage_start = 0.0

        run_sim = st.button("Run Physics Simulation & AI Health Diagnosis", type="primary", use_container_width=True)

        # Store simulation results in session state
        if run_sim or "sim_data" not in st.session_state:
            with st.spinner("Executing dynamic modal simulation and deep learning inference..."):
                sim_engine = BridgeSimulator(sampling_rate=100.0)
                scenario = BridgeDamageScenario(
                    damage_type=dmg_type_code,
                    location_m=damage_loc,
                    severity=damage_sev,
                    start_time_s=damage_start,
                )
                sim_res = sim_engine.simulate(
                    duration_s=sim_duration,
                    traffic_intensity=traffic_density,
                    mean_vehicle_speed_kmh=traffic_speed,
                    ambient_temp_c=ambient_temp,
                    wind_turbulence_level=wind_level,
                    noise_level_g=sensor_noise / 1000.0,
                    damage_scenario=scenario,
                    random_seed=42,
                )
                # Run AI pipeline
                times, assessments, reconstructions = pipeline.process_continuous_signal(sim_res["accelerations"])
                
                st.session_state["sim_res"] = sim_res
                st.session_state["ai_times"] = times
                st.session_state["ai_assessments"] = assessments
                st.session_state["ai_reconstructions"] = reconstructions
                st.session_state["damage_loc"] = damage_loc if dmg_type_code != "none" else None

        sim_res = st.session_state["sim_res"]
        assessments = st.session_state["ai_assessments"]
        ai_times = st.session_state["ai_times"]
        damage_site = st.session_state.get("damage_loc", None)

        st.plotly_chart(render_bridge_schematic(damage_loc=damage_site), use_container_width=True)

        # Vibration Waveform Visualizer
        st.subheader("Multi-Channel Vibration Sensor Telemetry")
        selected_sensor_idx = st.selectbox(
            "Select Sensor Station for Waveform & Spectral Inspection:",
            options=list(range(len(sim_res["sensor_names"]))),
            format_func=lambda idx: f"{sim_res['sensor_names'][idx]} - {pipeline.simulator.sensors[idx].description} ({pipeline.simulator.sensors[idx].location_m}m)",
            index=4,  # Midspan 2
        )

        sensor_name = sim_res["sensor_names"][selected_sensor_idx]
        raw_signal = sim_res["accelerations"][selected_sensor_idx]
        t = sim_res["time"]

        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=t, y=raw_signal, mode="lines", name=f"{sensor_name} Raw Acc", line=dict(color="#3b82f6", width=1.2)))
            if dmg_type_code != "none" and damage_sev > 0:
                fig_wave.add_vline(x=damage_start, line_width=2, line_dash="dash", line_color="#ef4444", annotation_text="Damage Onset", annotation_position="top left", annotation_font=dict(color="#FFFFFF"))
            fig_wave.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Acceleration (m/s²)",
                margin=dict(l=20, r=20, t=35, b=20),
            )
            apply_chart_theme(fig_wave, height=300, title=f"Acceleration Response: {sensor_name} ({pipeline.simulator.sensors[selected_sensor_idx].description})")
            st.plotly_chart(fig_wave, use_container_width=True)

        with col_w2:
            # Frequency spectrum
            fft_data = compute_fft_spectrum(raw_signal, fs=sim_res["sampling_rate"])
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=fft_data["frequencies"][:120], y=fft_data["magnitude"][:120], mode="lines", line=dict(color="#f59e0b", width=1.5), name="FFT Spectrum"))
            fig_fft.update_layout(
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                margin=dict(l=20, r=20, t=35, b=20),
            )
            apply_chart_theme(fig_fft, height=300, title=f"FFT Spectrum: {sensor_name}")
            st.plotly_chart(fig_fft, use_container_width=True)

    # =========================================================================
    # MODULE 2: AI ANOMALY DETECTION & HEALTH SCORING
    # =========================================================================
    elif menu == "Anomaly Detection & Health Scoring":
        st.header("Deep Learning Anomaly Detection & Structural Health Scoring")
        st.markdown(
            "Autoencoder reconstruction analysis, continuous Structural Health Index (SHI) gauge, "
            "and spatial damage localization across bridge sensor nodes."
        )

        if "ai_assessments" not in st.session_state:
            st.info("No active simulation found. Running baseline diagnosis...")
            sim_engine = BridgeSimulator(sampling_rate=100.0)
            sim_res = sim_engine.simulate(duration_s=45.0, random_seed=42)
            times, assessments, reconstructions = pipeline.process_continuous_signal(sim_res["accelerations"])
            st.session_state["sim_res"] = sim_res
            st.session_state["ai_times"] = times
            st.session_state["ai_assessments"] = assessments
            st.session_state["ai_reconstructions"] = reconstructions

        assessments = st.session_state["ai_assessments"]
        ai_times = st.session_state["ai_times"]
        sim_res = st.session_state["sim_res"]

        # Aggregate Statistics
        recent_assessments = assessments[-15:] if len(assessments) >= 15 else assessments
        latest_shi = np.mean([a.health_index for a in recent_assessments])
        total_anomalies = sum(1 for a in assessments if a.is_anomaly)
        anomaly_ratio = total_anomalies / max(1, len(assessments))
        latest_status = assessments[-1].status

        # KPI Dashboard
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            status_colors = {"Healthy": "normal", "Advisory": "inverse", "Warning": "off", "Critical": "inverse"}
            st.metric(
                label="Structural Health Index (SHI)",
                value=f"{latest_shi:.1f}%",
                delta=f"Status: {latest_status}",
                delta_color="normal" if latest_shi >= 80 else ("off" if latest_shi >= 50 else "inverse"),
            )
        with kpi2:
            st.metric(
                label="Anomaly Detection Rate",
                value=f"{anomaly_ratio:.1%}",
                delta=f"{total_anomalies} of {len(assessments)} windows",
                delta_color="inverse" if anomaly_ratio > 0.15 else "normal",
            )
        with kpi3:
            max_err = max(a.anomaly_score for a in assessments)
            st.metric(
                label="Peak Reconstruction Error",
                value=f"{max_err:.4f}",
                delta=f"Threshold: {assessments[0].threshold:.4f}",
                delta_color="inverse" if max_err > assessments[0].threshold else "normal",
            )
        with kpi4:
            most_damaged = assessments[-1].most_affected_sensor
            st.metric(
                label="Damage Hotspot (Spatial)",
                value=f"{most_damaged}",
                delta="Primary Anomaly Center",
            )

        st.divider()

        # Structural Health Index Gauge & Reconstruction Error Timeline
        g_col1, g_col2 = st.columns([1, 2])
        with g_col1:
            st.subheader("Structural Health Gauge")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=latest_shi,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "SHI Score (0 - 100%)", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                        "bar": {"color": "#1f77b4"},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "gray",
                        "steps": [
                            {"range": [0, 50], "color": "#ffcccc"},
                            {"range": [50, 75], "color": "#ffe6cc"},
                            {"range": [75, 90], "color": "#ffffcc"},
                            {"range": [90, 100], "color": "#ccffcc"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            st.subheader("Reconstruction Error vs Calibrated Anomaly Threshold")
            fig_err = go.Figure()
            err_vals = [a.anomaly_score for a in assessments]
            thresh_vals = [a.threshold for a in assessments]
            is_anom = [a.is_anomaly for a in assessments]

            fig_err.add_trace(go.Scatter(x=ai_times, y=err_vals, mode="lines+markers", name="Reconstruction MSE", line=dict(color="#1f77b4", width=2)))
            fig_err.add_trace(go.Scatter(x=ai_times, y=thresh_vals, mode="lines", name="Dynamic Threshold τ", line=dict(color="red", width=2, dash="dash")))

            # Highlight anomalies
            anom_x = [t for t, anom in zip(ai_times, is_anom) if anom]
            anom_y = [e for e, anom in zip(err_vals, is_anom) if anom]
            if len(anom_x) > 0:
                fig_err.add_trace(go.Scatter(x=anom_x, y=anom_y, mode="markers", name="Flagged Anomaly", marker=dict(color="red", size=8, symbol="circle-open-dot")))

            fig_err.update_layout(
                title="Continuous Anomaly Telemetry",
                xaxis_title="Time (seconds)",
                yaxis_title="Autoencoder Reconstruction MSE",
                height=280,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # Damage Localization Heatmap across all 8 Sensors
        st.subheader("Spatial Damage Localization & Attribution")
        st.markdown("Attribution of reconstruction error across individual sensor stations identifies the structural fault origin:")

        # Average attribution across windows
        sensor_names = sim_res["sensor_names"]
        avg_attrib = {s: 0.0 for s in sensor_names}
        for a in assessments:
            for s, val in a.sensor_attribution.items():
                avg_attrib[s] += val / len(assessments)

        df_attrib = pd.DataFrame({
            "Sensor": list(avg_attrib.keys()),
            "Attribution Score": list(avg_attrib.values()),
        })

        fig_bar = px.bar(
            df_attrib,
            x="Sensor",
            y="Attribution Score",
            color="Attribution Score",
            color_continuous_scale="Reds",
            title="Sensor Anomaly Share (Higher indicates localized stiffness loss or micro-crack)",
        )
        fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # MODULE 3: BRIDGE DIGITAL TWIN & FREQUENCIES
    # =========================================================================
    elif menu == "Bridge Digital Twin & Frequencies":
        st.header("Bridge Digital Twin & Modal Frequency Analysis")
        st.markdown(
            "Structural specifications of the 220m 3-span continuous steel-concrete girder bridge, "
            "sensor deployment topology, and analytical mode shapes."
        )

        sim = BridgeSimulator()
        st.plotly_chart(render_bridge_schematic(), use_container_width=True)

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.subheader("Sensor Station Deployment")
            sensor_df = pd.DataFrame([
                {
                    "Sensor ID": s.id,
                    "Location (m)": f"{s.location_m:.1f} m",
                    "Span": f"Span {s.span_index + 1}",
                    "Description": s.description,
                    "Sensor Type": "Distributed Accelerometer Channel",
                }
                for s in sim.sensors
            ])
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)

        with col_t2:
            st.subheader("Natural Modal Frequencies (Healthy Baseline)")
            modes_df = pd.DataFrame({
                "Mode #": [f"Mode {i+1}" for i in range(len(sim.nominal_freqs))],
                "Frequency (Hz)": sim.nominal_freqs,
                "Damping Ratio (ζ)": [f"{z*100:.1f}%" for z in sim.damping_ratios],
                "Description": [
                    "1st Symmetric Vertical Bending",
                    "1st Asymmetric Vertical Bending",
                    "2nd Symmetric Vertical Bending",
                    "1st Torsional / Higher Vertical",
                    "Higher Order Complex Mode",
                ],
            })
            st.dataframe(modes_df, use_container_width=True, hide_index=True)

        # Plot Analytical Mode Shapes
        st.subheader("Theoretical Mode Shapes along Bridge Deck")
        x_pts = np.linspace(0, sim.L, 300)
        fig_modes = go.Figure()
        for m in range(3):
            phi = [float(sim._mode_shape(m, np.array([x]))[0]) for x in x_pts]
            fig_modes.add_trace(go.Scatter(x=x_pts, y=phi, mode="lines", name=f"Mode {m+1} ({sim.nominal_freqs[m]} Hz)"))

        # Mark Piers
        fig_modes.add_vline(x=60, line_width=1.5, line_dash="dash", line_color="gray", annotation_text="Pier 1 (60m)")
        fig_modes.add_vline(x=160, line_width=1.5, line_dash="dash", line_color="gray", annotation_text="Pier 2 (160m)")
        fig_modes.update_layout(
            title="Continuous Beam Eigenmode Shapes",
            xaxis_title="Bridge Longitudinal Position (m)",
            yaxis_title="Normalized Modal Displacement φ(x)",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_modes, use_container_width=True)

    # =========================================================================
    # MODULE 4: MODEL ARCHITECTURE & PERFORMANCE
    # =========================================================================
    elif menu == "Model Architecture & Performance":
        st.header("Deep Learning Architecture & Evaluation Metrics")
        st.markdown(
            "Detailed inspection of the **Conv1D-BiLSTM Autoencoder** network parameters, "
            "unsupervised training configuration, and benchmark classification metrics."
        )

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            st.subheader("Network Architecture")
            st.markdown(
                """
                - **Encoder**:
                  - Conv1D Block 1: `Conv1d(8 -> 32, k=7, s=2)` + BatchNorm + LeakyReLU + Dropout
                  - Conv1D Block 2: `Conv1d(32 -> 64, k=5, s=2)` + BatchNorm + LeakyReLU + Dropout
                  - Conv1D Block 3: `Conv1d(64 -> 96, k=3, s=2)` + BatchNorm + LeakyReLU
                  - Bidirectional LSTM: `Hidden Size = 64, Layers = 1` (Captures forward & backward modal reverberations)
                  - Latent Projection: Dense layer compressing to `Latent Dimension = 48`
                - **Decoder**:
                  - Latent Unprojection: Dense layer expanding to `128`
                  - Bidirectional LSTM Decoder: `Hidden Size = 48, Layers = 1`
                  - ConvTranspose1D Block 1: `ConvTranspose1d(96 -> 64, k=4, s=2)`
                  - ConvTranspose1D Block 2: `ConvTranspose1d(64 -> 32, k=4, s=2)`
                  - ConvTranspose1D Block 3: `ConvTranspose1d(32 -> 8, k=4, s=2)`
                - **Loss Function**: Smooth L1 / Huber Loss ($\beta=0.05$) with AdamW optimizer & Cosine Annealing.
                """
            )
            # Total parameter count
            total_params = sum(p.numel() for p in pipeline.model.parameters())
            trainable_params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad)
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")

        with col_a2:
            st.subheader("Held-Out Synthetic Benchmark")
            metrics_path = "artifacts/evaluation_metrics.json"
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, "r", encoding="utf-8") as f:
                    benchmark = json.load(f)

                pct = lambda x: f"{100.0 * x:.1f}%"
                eval_metrics = pd.DataFrame({
                    "Metric": [
                        "ROC-AUC",
                        "PR-AUC (Average Precision)",
                        "F1-Score",
                        "Detection Recall",
                        "Precision",
                        "False Alarm Rate (FAR)",
                    ],
                    "Value": [
                        f"{benchmark['roc_auc']:.3f}",
                        f"{benchmark['pr_auc']:.3f}",
                        f"{benchmark['f1_score']:.3f}",
                        pct(benchmark['recall']),
                        pct(benchmark['precision']),
                        pct(benchmark['false_alarm_rate']),
                    ],
                })
                st.dataframe(eval_metrics, use_container_width=True, hide_index=True)
                st.caption(
                    f"{benchmark['evaluation_set']['num_windows']} windows: "
                    f"{benchmark['evaluation_set']['healthy_windows']} healthy + "
                    f"{benchmark['evaluation_set']['anomalous_windows']} anomalous. "
                    "Metrics are computed from the saved checkpoint on held-out synthetic simulation runs."
                )
            else:
                st.info(
                    "Benchmark metrics have not been generated yet. Run `python quickstart.py` "
                    "once to create artifacts/evaluation_metrics.json."
                )

            st.markdown(
                """
                > [!NOTE]
                > The model is trained purely on **unsupervised healthy baseline data**.
                > The benchmark uses separate synthetic simulation runs generated after training.
                > These results demonstrate performance within the synthetic environment and are not
                > validation on real-world bridge deployments.
                """
            )

        st.divider()
        st.subheader("Retrain Baseline Model")
        with st.expander("Retrain with Custom Hyperparameters"):
            train_epochs = st.slider("Training Epochs:", 5, 40, 15)
            train_dur = st.slider("Simulation Training Duration (s):", 60, 300, 180, step=30)
            if st.button("Start Retraining", type="secondary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def callback(epoch, total, t_loss, v_loss):
                    progress_bar.progress(epoch / total)
                    status_text.text(f"Epoch {epoch}/{total} - Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

                res = pipeline.train_on_baseline(
                    duration_s=train_dur,
                    epochs=train_epochs,
                    progress_callback=callback,
                )
                pipeline.save_checkpoint(CHECKPOINT_PATH)
                st.success(f"Retraining Complete! Best Val Loss: {res['best_val_loss']:.5f} | New Threshold: {res['calibrated_threshold']:.5f}")

    # =========================================================================
    # MODULE 5: CUSTOM SENSOR DATA CSV
    # =========================================================================
    elif menu == "Custom Sensor Data":
        st.header("Custom Sensor Data Analysis & Upload")
        st.markdown(
            "Upload external bridge acceleration datasets (CSV format) or custom finite element simulation output."
        )

        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_file = st.file_uploader(
                "Upload Bridge Accelerometer CSV:",
                type=["csv"],
                help="CSV should have rows as timesteps and columns as acceleration channels.",
            )

        with col_u2:
            st.markdown("**Sample Benchmark Files:**")
            if os.path.exists("data/sample_healthy.csv"):
                with open("data/sample_healthy.csv", "rb") as f:
                    st.download_button("Download Sample Healthy CSV", f.read(), "sample_healthy.csv", "text/csv")
            if os.path.exists("data/sample_damaged.csv"):
                with open("data/sample_damaged.csv", "rb") as f:
                    st.download_button("Download Sample Damaged CSV", f.read(), "sample_damaged.csv", "text/csv")

        # Load either uploaded file or default sample
        target_csv = None
        if uploaded_file is not None:
            target_csv = uploaded_file
        elif os.path.exists("data/sample_damaged.csv"):
            st.info("No file uploaded yet. Showing analysis on pre-generated `sample_damaged.csv` benchmark.")
            target_csv = "data/sample_damaged.csv"

        if target_csv is not None:
            try:
                if hasattr(target_csv, "seek"):
                    target_csv.seek(0)
                acc_matrix, sensor_cols, time_vec = load_sensor_csv(target_csv)
                st.success(f"Successfully loaded dataset with {len(sensor_cols)} sensor channels and {acc_matrix.shape[1]} timesteps.")

                # If sensor count doesn't match 8, pad or slice for model compatibility
                if acc_matrix.shape[0] != 8:
                    st.warning(f"Dataset has {acc_matrix.shape[0]} channels. Model expects 8 channels. Adjusting channel dimension...")
                    if acc_matrix.shape[0] < 8:
                        pad = np.zeros((8 - acc_matrix.shape[0], acc_matrix.shape[1]))
                        acc_matrix = np.vstack([acc_matrix, pad])
                    else:
                        acc_matrix = acc_matrix[:8, :]

                # Check minimum required signal length for sliding windows
                if acc_matrix.shape[1] < pipeline.window_size:
                    st.error(
                        f"Dataset has {acc_matrix.shape[1]} timesteps, but the model requires at least "
                        f"{pipeline.window_size} samples ({pipeline.window_size / pipeline.fs:.2f}s) for evaluation windows."
                    )
                else:
                    # Run Inference
                    with st.spinner("Analyzing custom vibration stream..."):
                        times, assessments, recs = pipeline.process_continuous_signal(acc_matrix, sensor_names=sensor_cols[:8])

                    if assessments and len(assessments) > 0:
                        # Results Summary
                        avg_shi = float(np.mean([a.health_index for a in assessments]))
                        anom_count = sum(1 for a in assessments if a.is_anomaly)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted Health Index", f"{avg_shi:.1f}%")
                        c2.metric(
                            "Anomalous Windows",
                            f"{anom_count} / {len(assessments)}",
                            delta=f"{anom_count/len(assessments):.1%}",
                            delta_color="inverse" if anom_count > 0 else "normal",
                        )
                        c3.metric("Most Damaged Sensor", assessments[-1].most_affected_sensor)

                        # Time series plot
                        fig_custom = go.Figure()
                        fig_custom.add_trace(
                            go.Scatter(
                                x=times,
                                y=[a.anomaly_score for a in assessments],
                                mode="lines+markers",
                                name="Reconstruction Error",
                                line=dict(color="#3b82f6"),
                            )
                        )
                        fig_custom.add_trace(
                            go.Scatter(
                                x=times,
                                y=[a.threshold for a in assessments],
                                mode="lines",
                                name="Threshold",
                                line=dict(color="#ef4444", dash="dash"),
                            )
                        )
                        fig_custom = apply_chart_theme(fig_custom, height=280, title="Reconstruction Error Timeline")
                        fig_custom.update_layout(xaxis_title="Time (s)", yaxis_title="Error MSE")
                        st.plotly_chart(fig_custom, use_container_width=True)

                        # Export diagnosis report
                        report_df = pd.DataFrame([
                            {
                                "Window Time (s)": round(t, 2),
                                "Reconstruction Error": round(a.anomaly_score, 5),
                                "Anomaly Flag": a.is_anomaly,
                                "Health Index (%)": a.health_index,
                                "Status": a.status,
                                "Most Damaged Sensor": a.most_affected_sensor,
                            }
                            for t, a in zip(times, assessments)
                        ])
                        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Export Health Diagnosis Report (CSV)", csv_bytes, "bridge_health_report.csv", "text/csv")
                    else:
                        st.warning("Insufficient signal duration to extract evaluation windows.")

            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")


if __name__ == "__main__":
    main()
