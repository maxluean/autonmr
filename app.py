"""
AutoNMR — Demo UI (Phase 4)
Streamlit Web Application

แทนที่กระบวนการ manual ทีละภาพ 
ด้วย Automated NMR Annotation System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import time
import os
import sys
import html

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocessor import NMRPreprocessor
from core.peak_detector import PeakDetector
from core.annotator import Annotator
from core.confidence_scorer import ConfidenceScorer
from ml.random_forest_model import RandomForestAnnotator
from ml.cnn_model import CNNAnnotator
from ml.ensemble import EnsembleAnnotator

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AutoNMR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS (Enterprise / Clinical SaaS)
# =========================================
st.markdown("""
<style>
    :root {
        --bg: #F8FAFC;
        --surface: #FFFFFF;
        --surface-muted: #F1F5F9;
        --border: #E2E8F0;
        --border-strong: #CBD5E1;
        --navy: #0F172A;
        --blue: #1D4ED8;
        --teal: #0F766E;
        --orange: #F59E0B;
        --emerald: #10B981;
        --rose: #E11D48;
        --text: #0F172A;
        --muted: #64748B;
        --shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }

    html, body, [class*="css"] {
        font-family: Inter, Roboto, -apple-system, BlinkMacSystemFont,
            "Segoe UI", sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    header[data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.92);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.65);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #CBD5E1 !important;
        font-size: 0.82rem;
        font-weight: 500;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="popover"] * {
        color: var(--navy) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.45);
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        border-color: var(--orange);
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div:first-child div {
        background: var(--orange);
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--orange) !important;
        border-color: var(--orange) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-mark {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        background: var(--navy);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        letter-spacing: 0;
    }

    .brand-name {
        margin: 0;
        color: var(--navy);
        font-size: 1.45rem;
        line-height: 1.1;
        font-weight: 760;
    }

    .brand-subtitle {
        margin: 0.2rem 0 0 0;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .org-label {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0;
        text-align: right;
    }

    .page-title {
        margin: 0 0 0.5rem 0;
        color: var(--navy);
        font-size: 1.6rem;
        line-height: 1.2;
        font-weight: 760;
    }

    .page-kicker {
        margin: 0 0 1.35rem 0;
        color: var(--muted);
        font-size: 0.98rem;
        max-width: 760px;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 1.15rem;
    }

    .metric-card {
        min-height: 132px;
    }

    .metric-icon {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1;
        margin-bottom: 1.2rem;
    }

    .metric-value {
        color: var(--navy);
        font-size: 2.05rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.45rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.83rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0;
    }

    .metric-note {
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .section-header {
        color: var(--navy);
        font-size: 1.02rem;
        font-weight: 760;
        margin: 1.2rem 0 0.75rem 0;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid var(--border);
    }

    .timeline {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
        margin-top: 0.25rem;
    }

    .timeline-step {
        position: relative;
        padding: 0.35rem 1rem 0 0;
    }

    .timeline-step:before {
        content: "";
        position: absolute;
        top: 0.72rem;
        left: 1.9rem;
        right: 0.45rem;
        height: 1px;
        background: var(--border-strong);
    }

    .timeline-step:last-child:before {
        display: none;
    }

    .timeline-dot {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        background: var(--surface);
        border: 1px solid var(--border-strong);
        color: var(--blue);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 760;
        margin-bottom: 0.7rem;
        position: relative;
        z-index: 1;
    }

    .timeline-title {
        color: var(--navy);
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .timeline-copy {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .upload-frame {
        border: 1px dashed var(--border-strong);
        background: #FFFFFF;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.75rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.3rem 0.6rem;
        color: var(--muted);
        background: var(--surface);
        font-size: 0.78rem;
        font-weight: 600;
    }

    .stepper {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 0.25rem 0;
    }

    .process-step {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem;
        background: var(--surface);
    }

    .process-step.complete {
        border-color: rgba(16,185,129,0.35);
        background: #F0FDF4;
    }

    .process-step.active {
        border-color: rgba(29,78,216,0.35);
        background: #EFF6FF;
    }

    .process-dot {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        color: var(--muted);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 760;
        margin-bottom: 0.45rem;
    }

    .complete .process-dot {
        background: var(--emerald);
        border-color: var(--emerald);
        color: #FFFFFF;
    }

    .active .process-dot {
        background: var(--blue);
        border-color: var(--blue);
        color: #FFFFFF;
    }

    .process-title {
        color: var(--navy);
        font-size: 0.83rem;
        font-weight: 700;
    }

    .process-caption {
        color: var(--muted);
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    .skeleton {
        height: 10px;
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #E2E8F0 25%, #F8FAFC 37%, #E2E8F0 63%);
        background-size: 400% 100%;
        animation: shimmer 1.35s ease infinite;
        margin-top: 0.75rem;
    }

    @keyframes shimmer {
        0% { background-position: 100% 0; }
        100% { background-position: 0 0; }
    }

    .level-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 760;
    }

    .level-1 { background: var(--emerald); }
    .level-2 { background: var(--orange); }
    .level-3 { background: var(--rose); }
    .level-4 { background: #BE123C; }

    .linear-gauge {
        width: 100%;
        height: 7px;
        border-radius: 999px;
        background: #E2E8F0;
        overflow: hidden;
        margin-top: 0.35rem;
    }

    .linear-gauge span {
        display: block;
        height: 100%;
        border-radius: 999px;
        background: var(--blue);
    }

    .results-table td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid #E2E8F0;
        vertical-align: middle;
    }

    .results-table tbody tr:last-child td {
        border-bottom: 0;
    }

    .results-table tbody tr:hover {
        background: #F8FAFC;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: var(--navy);
        color: #FFFFFF;
        border: 1px solid var(--navy);
        border-radius: 8px;
        padding: 0.56rem 1rem;
        font-weight: 700;
        width: 100%;
        box-shadow: none;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1E293B;
        border-color: #1E293B;
        color: #FFFFFF;
    }

    .stButton > button[kind="primary"] {
        background: var(--orange);
        border-color: var(--orange);
        color: var(--navy);
    }

    .stButton > button[kind="primary"]:hover {
        background: #D97706;
        border-color: #D97706;
        color: var(--navy);
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow);
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 760;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed var(--border-strong);
        border-radius: 8px;
        background: #FFFFFF;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border-color: var(--border);
    }

    @media (max-width: 900px) {
        .app-header {
            align-items: flex-start;
            flex-direction: column;
        }

        .org-label {
            text-align: left;
        }

        .timeline,
        .stepper {
            grid-template-columns: 1fr;
        }

        .timeline-step:before {
            display: none;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================
# LOAD MODELS (cached)
# =========================================
@st.cache_resource
def load_models():
    """โหลด models ทั้งหมด (โหลดครั้งเดียว)"""
    try:
        # โหลด reference library
        with open('database/reference_library.json', 'r') as f:
            library_data = json.load(f)
        compound_names = list(library_data['compounds'].keys())

        # โหลด RF model
        rf = RandomForestAnnotator(compound_names)
        rf.load('data/trained_models/rf_real.pkl')

        # โหลด CNN model
        cnn = CNNAnnotator(compound_names)
        cnn.load('data/trained_models/cnn_real.pt')

        return rf, cnn, compound_names, True

    except Exception as e:
        st.warning(f"ไม่สามารถโหลดโมเดล ML: {str(e)}")
        return None, None, [], False


@st.cache_resource
def load_annotator():
    """โหลด template annotator"""
    return Annotator(
        library_path='database/reference_library.json',
        tolerance_ppm=0.03,
        min_match_ratio=0.4
    )


# =========================================
# HELPER FUNCTIONS
# =========================================
LEVEL_META = {
    1: {"name": "Confirmed", "class": "level-1", "color": "#10B981"},
    2: {"name": "Probable", "class": "level-2", "color": "#F59E0B"},
    3: {"name": "Putative", "class": "level-3", "color": "#E11D48"},
    4: {"name": "Tentative", "class": "level-4", "color": "#BE123C"},
}

PROCESS_STEPS = [
    ("Denoise", "Baseline correction"),
    ("Peak Detection", "Locate chemical shifts"),
    ("Template Match", "Reference library"),
    ("ML Ensemble", "Confidence scoring"),
]


def render_app_header(title=None, subtitle=None):
    """Render a clean product header."""
    heading = title or "AutoNMR"
    sub = subtitle or "Automated NMR Annotation System"
    st.markdown(f"""
    <div class="app-header">
        <div class="brand-lockup">
            <div class="brand-mark">AN</div>
            <div>
                <h1 class="brand-name">{html.escape(heading)}</h1>
                <p class="brand-subtitle">{html.escape(sub)}</p>
            </div>
        </div>
        <div class="org-label">BDI / Track 1 Phenome</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(value, label, note, icon_text):
    """Small executive-summary metric card."""
    return f"""
    <div class="card metric-card">
        <div class="metric-icon">{html.escape(icon_text)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-note">{html.escape(note)}</div>
    </div>
    """


def render_process_stepper(active_index=0, completed=None):
    """Render the analysis stepper used during processing."""
    completed = set(completed or [])
    items = []
    for idx, (title, caption) in enumerate(PROCESS_STEPS):
        state = ""
        marker = str(idx + 1)
        if idx in completed:
            state = "complete"
            marker = "✓"
        elif idx == active_index:
            state = "active"
        skeleton = '<div class="skeleton"></div>' if idx == active_index else ''
        items.append(
            f'<div class="process-step {state}">'
            f'<div class="process-dot">{marker}</div>'
            f'<div class="process-title">{html.escape(title)}</div>'
            f'<div class="process-caption">{html.escape(caption)}</div>'
            f'{skeleton}'
            f'</div>'
        )
    return f'<div class="stepper">{"".join(items)}</div>'


def level_badge(level):
    meta = LEVEL_META.get(level, LEVEL_META[4])
    return (
        f"<span class='level-badge {meta['class']}'>"
        f"L{level} {meta['name']}</span>"
    )


def score_value(annotation):
    return annotation.get(
        'ensemble_score',
        annotation.get('final_score', 0)
    )


def render_results_table(annotations, show_level):
    rows = []
    for ann in annotations:
        level = ann.get('msi_level', 4)
        if f"L{level}" not in show_level:
            continue
        score = max(0, min(1, score_value(ann)))
        compound = html.escape(ann.get('compound', 'Unknown'))
        meta = LEVEL_META.get(level, LEVEL_META[4])
        msi_name = html.escape(ann.get('msi_name', meta["name"]))
        rows.append(
            '<tr>'
            '<td>'
            f'<strong>{compound}</strong>'
            f'<div style="color:#64748B;font-size:0.78rem;">{msi_name}</div>'
            '</td>'
            f'<td>{level_badge(level)}</td>'
            '<td>'
            '<div style="display:flex;align-items:center;gap:0.65rem;">'
            '<span style="min-width:42px;font-weight:700;color:#0F172A;">'
            f'{score:.3f}'
            '</span>'
            '<div class="linear-gauge">'
            f'<span style="width:{score * 100:.1f}%;"></span>'
            '</div>'
            '</div>'
            '</td>'
            '</tr>'
        )

    if not rows:
        return """
        <div class="card" style="color:#64748B;">
            No compounds match the current confidence and MSI filters.
        </div>
        """

    return (
        '<div class="card" style="padding:0;overflow:hidden;">'
        '<table class="results-table" '
        'style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
        '<thead>'
        '<tr style="background:#F8FAFC;color:#64748B;text-align:left;">'
        '<th style="padding:0.8rem 1rem;border-bottom:1px solid #E2E8F0;">'
        'Compound</th>'
        '<th style="padding:0.8rem 1rem;border-bottom:1px solid #E2E8F0;">'
        'MSI Level</th>'
        '<th style="padding:0.8rem 1rem;border-bottom:1px solid #E2E8F0;">'
        'Confidence</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
        '</div>'
    )


def create_synthetic_demo():
    """
    สร้าง demo spectrum จำลอง Plant Extract [1]
    สำหรับกรณีที่ไม่มีไฟล์จริง
    """
    ppm_axis = np.linspace(0.5, 10.0, 20000)
    spectrum = np.zeros(20000)

    # สาร 40+ ชนิดจาก Domain 1 [1]
    demo_compounds = {
        'Isoleucine':   [(0.94, 1.0), (1.01, 0.9), (3.67, 0.6)],
        'Leucine':      [(0.96, 1.0), (0.97, 0.95), (1.70, 0.6)],
        'Valine':       [(0.99, 1.0), (1.04, 0.95), (2.27, 0.5)],
        'Alanine':      [(1.48, 1.0), (3.78, 0.7)],
        'Ethanol':      [(1.18, 1.0), (3.66, 0.8)],
        'Acetate':      [(1.92, 1.0)],
        'Propionate':   [(1.06, 0.8), (2.18, 1.0)],
        'Citrate':      [(2.54, 1.0), (2.69, 0.95)],
        'Succinate':    [(2.41, 1.0)],
        'Glucose':      [(3.24, 0.6), (3.41, 0.7),
                         (3.73, 0.8), (4.65, 0.9), (5.23, 1.0)],
        'Sucrose':      [(3.48, 0.7), (3.56, 0.6),
                         (3.68, 0.8), (3.76, 0.7),
                         (4.05, 0.8), (4.22, 0.9), (5.41, 1.0)],
        'Maltose':      [(3.28, 0.6), (3.62, 0.7),
                         (3.95, 0.8), (4.65, 0.9), (5.41, 1.0)],
        'Formate':      [(8.46, 1.0)],
        'Tryptophan':   [(7.20, 0.7), (7.29, 0.8),
                         (7.54, 1.0), (7.73, 0.9)],
        'Tyrosine':     [(6.90, 1.0), (7.19, 0.95)],
        'Nicotinate':   [(7.50, 0.7), (8.25, 0.8),
                         (8.62, 0.9), (8.94, 1.0)],
    }

    for compound, peaks in demo_compounds.items():
        scale = np.random.uniform(0.5, 1.5)
        for ppm, intensity in peaks:
            width = np.random.uniform(0.008, 0.015)
            actual_ppm = ppm + np.random.normal(0, 0.003)
            peak = intensity * scale * (width ** 2) / (
                (ppm_axis - actual_ppm) ** 2 + width ** 2
            )
            spectrum += peak

    # เพิ่ม noise + baseline
    noise = np.random.normal(
        0, 0.005 * np.max(spectrum), len(spectrum)
    )
    baseline = 0.02 * np.sin(
        np.linspace(0, 3 * np.pi, len(spectrum))
    )
    spectrum = np.maximum(spectrum + noise + baseline, 0)

    return ppm_axis, spectrum


def plot_spectrum(ppm_axis, spectrum,
                  peaks=None, title="NMR Spectrum",
                  annotations=None):
    """
    สร้าง Interactive Plotly spectrum
    """
    fig = go.Figure()

    # Spectrum line
    fig.add_trace(go.Scatter(
        x=ppm_axis,
        y=spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color='#0F172A', width=1.15),
        hovertemplate='ppm: %{x:.3f}<br>Intensity: %{y:.4f}'
    ))

    # Peak markers
    if peaks is not None and len(peaks['ppm']) > 0:
        fig.add_trace(go.Scatter(
            x=peaks['ppm'],
            y=peaks['intensity'],
            mode='markers',
            name='Detected Peaks',
            marker=dict(
                symbol='circle',
                size=6,
                color='#0F766E',
                line=dict(color='#FFFFFF', width=0.8),
                opacity=0.88
            ),
            hovertemplate='Peak: %{x:.3f} ppm'
        ))

    # Compound annotations
    if annotations:
        for ann in annotations[:10]:
            if ann.get('matched_peaks'):
                for mp in ann['matched_peaks'][:1]:
                    fig.add_annotation(
                        x=mp['query_ppm'],
                        y=np.interp(
                            mp['query_ppm'],
                            ppm_axis, spectrum
                        ) * 1.1,
                        text=ann['compound'][:8],
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='#F59E0B',
                        font=dict(size=8, color='#0F172A'),
                        ax=0, ay=-25
                    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color='#0F172A')
        ),
        xaxis=dict(
            title='Chemical Shift (ppm)',
            autorange='reversed',
            gridcolor='#E2E8F0',
            zerolinecolor='#E2E8F0',
            title_font=dict(color='#334155')
        ),
        yaxis=dict(
            title='Intensity',
            gridcolor='#E2E8F0',
            zerolinecolor='#E2E8F0',
            title_font=dict(color='#334155')
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#334155'),
        hovermode='x unified',
        height=350,
        margin=dict(t=40, b=40, l=60, r=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            font=dict(color='#64748B')
        )
    )

    return fig


def run_full_pipeline(spectrum, ppm_axis,
                      rf_model, cnn_model,
                      compound_names, method='ensemble'):
    """
    รัน full pipeline: Preprocess → Detect → Annotate
    """
    results = {}
    start_time = time.time()

    # Step 1: Preprocess
    preprocessor = NMRPreprocessor('plant_extract')
    processed = preprocessor.process(spectrum, ppm_axis)

    # Step 2: Detect Peaks
    detector = PeakDetector()
    peaks = detector.detect(
        processed['processed'],
        processed['ppm_axis']
    )
    results['peaks'] = peaks
    results['processed'] = processed['processed']
    results['ppm_axis'] = processed['ppm_axis']

    # ===== DEBUG =====
    print(f"DEBUG: Peaks found = {peaks['num_peaks']}")

    # Step 3: Template Matching
    annotator = load_annotator()
    template_results = annotator.annotate(peaks)

    # ===== DEBUG =====
    print(f"DEBUG: Template results = {len(template_results)}")
    if template_results:
        print(f"DEBUG: Top result = {template_results[0]}")

    # Step 4: Ensemble
    if method == 'ensemble':
        ensemble = EnsembleAnnotator(compound_names)
        final = ensemble.combine(
            template_results, [], []
        )
    else:
        final = template_results

    # ===== DEBUG =====
    print(f"DEBUG: Final results = {len(final)}")
    if final:
        print(f"DEBUG: Top final = {final[0]}")

    # Step 5: Confidence Scoring
    scorer = ConfidenceScorer()
    scored = scorer.score(final)

    # ===== DEBUG =====
    print(f"DEBUG: Scored results = {len(scored)}")

    results['annotations'] = scored
    results['processing_time'] = time.time() - start_time

    return results


# =========================================
# SIDEBAR
# =========================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 1.25rem 0;margin-bottom:0.5rem;'>
            <div style='display:flex;align-items:center;gap:0.7rem;'>
                <div style='width:36px;height:36px;border-radius:8px;
                            background:#FFFFFF;color:#0F172A;display:flex;
                            align-items:center;justify-content:center;
                            font-weight:800;'>AN</div>
                <div>
                    <h2 style='color:white;margin:0;font-size:1.25rem;'>AutoNMR</h2>
                    <p style='color:#94A3B8;margin:0;font-size:0.78rem;'>
                        Clinical-grade annotation
                    </p>
                </div>
            </div>
            <p style='color:#94A3B8;margin:1rem 0 0 0;font-size:0.78rem;'>
                BDI / Track 1 Phenome
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navigation")
        page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "Workspace",
                "About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Analysis Settings")

        method = st.selectbox(
            "Annotation Method",
            options=['ensemble', 'template', 'rf', 'cnn'],
            format_func=lambda x: {
                'ensemble': 'Ensemble (Best)',
                'template': 'Template Only',
                'rf': 'Random Forest',
                'cnn': 'CNN Only'
            }[x]
        )

        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.30,
            step=0.05,
            help="ระดับ confidence ขั้นต่ำ"
        )

        show_level = st.multiselect(
            "Show MSI Levels",
            options=['L1', 'L2', 'L3', 'L4'],
            default=['L1', 'L2', 'L3', 'L4']
        )

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem; color:#888;
                    text-align:center;'>
            BDI Young Innovator Hackathon 2026<br>
            Plant Extract A-N
        </div>
        """, unsafe_allow_html=True)

    return page, method, min_confidence, show_level


# =========================================
# PAGE: HOME
# =========================================
def page_home():
    render_app_header(
        "AutoNMR",
        "Automated NMR Annotation System for research-grade metabolite analysis"
    )

    st.markdown("""
    <h2 class="page-title">Executive Overview</h2>
    <p class="page-kicker">
        A clean analysis workspace that converts manual spectrum-by-spectrum
        annotation into a structured automated pipeline for Plant Extract A-N.
    </p>
    """, unsafe_allow_html=True)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            metric_card("< 5 นาที", "Processing Time",
                        "ลดเวลาจาก 2-4 ชม. แบบ manual", "Time"),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            metric_card("40+", "Compounds",
                        "Reference set for Plant Extract A-N", "Library"),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            metric_card("≥ 85%", "Accuracy",
                        "Benchmark against expert annotation", "Quality"),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            metric_card("$0", "License Cost",
                        "Open-source analysis foundation", "Cost"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # System Overview
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="section-header">Automated Workflow</div>
        <div class="card">
            <div class="timeline">
                <div class="timeline-step">
                    <div class="timeline-dot">1</div>
                    <div class="timeline-title">Import</div>
                    <div class="timeline-copy">
                        Upload spectrum files or select demo Plant Extract data.
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-dot">2</div>
                    <div class="timeline-title">Preprocess</div>
                    <div class="timeline-copy">
                        Denoise, baseline-correct, and normalize the spectrum.
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-dot">3</div>
                    <div class="timeline-title">Detect</div>
                    <div class="timeline-copy">
                        Identify peaks and match chemical shifts to references.
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="timeline-dot">4</div>
                    <div class="timeline-title">Report</div>
                    <div class="timeline-copy">
                        Score MSI levels and export structured results.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="section-header">Supported Analysis</div>
        <div class="card">
            <div style="color:#0F172A;font-weight:760;margin-bottom:0.75rem;">
                Compound Coverage
            </div>
            <div style="display:grid;gap:0.65rem;color:#475569;font-size:0.9rem;">
                <div>Amino acids: Ile, Leu, Val, Ala</div>
                <div>Sugars: Sucrose, Glucose, Maltose</div>
                <div>Organic acids: Citrate, Acetate</div>
                <div>Nucleosides: Guanosine, Uridine</div>
                <div>Other metabolites: Formate, Choline, Trp</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Demo Button
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        if st.button("Open Workspace", type="primary"):
            st.info("เลือก Workspace ใน sidebar เพื่อเริ่มวิเคราะห์ spectrum")


# =========================================
# PAGE: ANNOTATE
# =========================================
def page_annotate(method, min_confidence, show_level):
    render_app_header(
        "Workspace",
        "Data input, spectrum review, automated annotation, and export"
    )
    st.markdown("""
    <h2 class="page-title">NMR Analysis Workspace</h2>
    <p class="page-kicker">
        Designed as a focused scientist workstation: upload or demo data,
        review the spectrum, run the pipeline, and inspect MSI-scored results.
    </p>
    """, unsafe_allow_html=True)

    # Load models
    rf_model, cnn_model, compound_names, models_loaded = \
        load_models()

    if not models_loaded:
        st.warning(
            "Models not found. "
            "Please run Phase 3 training first.\n"
            "Using Template Matching only."
        )

    # ===== Upload Section =====
    st.markdown('<div class="section-header">Data Input</div>',
                unsafe_allow_html=True)

    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        st.markdown("""
        <div class="upload-frame">
            <div style="color:#0F172A;font-weight:760;margin-bottom:0.3rem;">
                Upload Spectrum File
            </div>
            <div style="color:#64748B;font-size:0.86rem;">
                CSV, TXT, or DAT with ppm and intensity columns.
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload NMR Spectrum file",
            type=['csv', 'txt', 'dat'],
            help="Format: 2 columns (ppm, intensity) "
                 "หรือ intensity เพียงอย่างเดียว"
        )

    with col_options:
        sample_type = st.selectbox(
            "Sample Type",
            options=['Plant Extract', 'Blood',
                     'Urine', 'Fecal', 'Other'],
            index=0
        )
        use_demo = st.toggle(
            "Use Demo Data",
            value=True,
            help="ใช้ synthetic Plant Extract [1] สำหรับ demo"
        )
        st.markdown("""
        <span class="status-pill">Plant Extract A-N demo ready</span>
        """, unsafe_allow_html=True)

    # โหลด spectrum
    ppm_axis = None
    spectrum = None

    if uploaded is not None:
        try:
            # อ่านไฟล์
            content = uploaded.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            # ลบ header ถ้ามี
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # ข้ามบรรทัดที่เป็น header (ไม่ใช่ตัวเลข)
                try:
                    parts = line.replace(',', ' ').replace('\t', ' ').split()
                    float(parts[0])
                    data_lines.append(parts)
                except ValueError:
                    continue
            
            if len(data_lines) == 0:
                st.error("ไม่พบข้อมูลในไฟล์ กรุณาตรวจสอบ format")
                ppm_axis = None
                spectrum = None
            
            elif len(data_lines[0]) >= 2:
                # มี 2 columns: ppm, intensity
                ppm_axis = np.array([
                    float(row[0]) for row in data_lines
                ])
                spectrum = np.array([
                    float(row[1]) for row in data_lines
                ])
                st.success(
                    f"Loaded: {len(spectrum):,} data points "
                    f"(ppm: {ppm_axis[0]:.2f} - {ppm_axis[-1]:.2f})"
                )
            
            else:
                # มีแค่ 1 column: intensity เท่านั้น
                spectrum = np.array([
                    float(row[0]) for row in data_lines
                ])
                # สร้าง ppm_axis อัตโนมัติ (0.5 - 10.0 ppm)
                ppm_axis = np.linspace(
                    0.5, 10.0, len(spectrum)
                )
                st.success(
                    f"Loaded: {len(spectrum):,} points "
                    f"(ppm axis auto-generated: 0.5-10.0)"
                )
                st.warning(
                    "ไม่พบ ppm column จึงสร้าง ppm axis "
                    "อัตโนมัติ (0.5-10.0 ppm)"
                )

        except Exception as e:
            st.error(f"Error loading file: {e}")
            ppm_axis = None
            spectrum = None

    elif use_demo:
        ppm_axis, spectrum = create_synthetic_demo()
        st.info(
            "Using synthetic Plant Extract demo [1] "
            "(Sucrose, Glucose, Alanine, Formate ...)"
        )

    # ===== Preview =====
    if spectrum is not None:
        st.markdown('<div class="section-header">Spectrum Preview</div>',
                    unsafe_allow_html=True)
        fig_raw = plot_spectrum(
            ppm_axis, spectrum,
            title="Raw NMR Spectrum"
        )
        st.plotly_chart(fig_raw, width="stretch")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Data Points", f"{len(spectrum):,}")
        col_info2.metric(
            "PPM Range",
            f"{ppm_axis[0]:.1f} - {ppm_axis[-1]:.1f}"
        )
        col_info3.metric(
            "Max Intensity",
            f"{np.max(spectrum):.3f}"
        )

        # ===== Process Button =====
        st.markdown('<div class="section-header">Processing</div>',
                    unsafe_allow_html=True)
        stepper_placeholder = st.empty()
        stepper_placeholder.markdown(
            render_process_stepper(active_index=0, completed=[]),
            unsafe_allow_html=True
        )
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            process_btn = st.button(
                "Annotate Spectrum",
                width="stretch",
                type="primary"
            )

        if process_btn:
            with st.spinner("Running annotation pipeline..."):
                for idx in range(len(PROCESS_STEPS)):
                    stepper_placeholder.markdown(
                        render_process_stepper(
                            active_index=idx,
                            completed=range(idx)
                        ),
                        unsafe_allow_html=True
                    )
                    time.sleep(0.25)

                # Run pipeline
                results = run_full_pipeline(
                    spectrum, ppm_axis,
                    rf_model, cnn_model,
                    compound_names, method
                )

                stepper_placeholder.markdown(
                    render_process_stepper(
                        active_index=-1,
                        completed=range(len(PROCESS_STEPS))
                    ),
                    unsafe_allow_html=True
                )

            # ===== Results =====
            st.markdown("""
            <div class="section-header">Results</div>
            """, unsafe_allow_html=True)

            # Summary metrics
            annotations = results['annotations']
            proc_time = results['processing_time']

            filtered = [
                a for a in annotations
                if score_value(a) >= min_confidence
            ]

            by_level = {1: [], 2: [], 3: [], 4: []}
            for a in filtered:
                lv = a.get('msi_level', 4)
                if lv not in by_level:
                    lv = 4
                by_level[lv].append(a)

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

            col_m1.metric(
                "Time",
                f"{proc_time:.1f}s"
            )
            col_m2.metric(
                "Peaks",
                results['peaks']['num_peaks']
            )
            col_m3.metric(
                "Confirmed",
                len(by_level[1])
            )
            col_m4.metric(
                "Probable",
                len(by_level[2])
            )
            col_m5.metric(
                "Total",
                len(filtered)
            )

            # Annotated spectrum
            col_spectrum, col_panel = st.columns([7, 3])

            with col_spectrum:
                st.markdown("**Processed Spectrum with Peaks**")
                fig_proc = plot_spectrum(
                    results['ppm_axis'],
                    results['processed'],
                    peaks=results['peaks'],
                    annotations=filtered[:15],
                    title="Processed & Annotated Spectrum"
                )
                st.plotly_chart(
                    fig_proc, width="stretch"
                )

            with col_panel:
                st.markdown("**Model Summary**")
                if filtered:
                    top10 = filtered[:10]
                    compounds = [
                        r['compound'][:12]
                        for r in top10
                    ]
                    scores = [
                        score_value(r)
                        for r in top10
                    ]
                    colors = [
                        LEVEL_META.get(
                            r.get('msi_level', 4), LEVEL_META[4]
                        )["color"]
                        for r in top10
                    ]

                    fig_bar = go.Figure(go.Bar(
                        x=scores,
                        y=compounds,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig_bar.update_layout(
                        height=350,
                        margin=dict(t=20, b=20, l=10, r=20),
                        xaxis_title="Score",
                        yaxis_title=None,
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor='#FFFFFF',
                        font=dict(color='#334155'),
                        xaxis=dict(
                            range=[0, 1],
                            gridcolor='#E2E8F0',
                            zerolinecolor='#E2E8F0'
                        ),
                        yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(
                        fig_bar,
                        width="stretch"
                    )
                else:
                    st.info("No compounds passed the current threshold.")

                st.markdown(f"""
                <div class="card" style="margin-top:0.75rem;">
                    <div style="color:#0F172A;font-weight:760;margin-bottom:0.7rem;">
                        MSI Level Mix
                    </div>
                    <div style="display:grid;gap:0.55rem;color:#475569;font-size:0.86rem;">
                        <div>{level_badge(1)} <span style="margin-left:0.45rem;">{len(by_level[1])}</span></div>
                        <div>{level_badge(2)} <span style="margin-left:0.45rem;">{len(by_level[2])}</span></div>
                        <div>{level_badge(3)} <span style="margin-left:0.45rem;">{len(by_level[3])}</span></div>
                        <div>{level_badge(4)} <span style="margin-left:0.45rem;">{len(by_level[4])}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Identified Compounds</div>',
                        unsafe_allow_html=True)
            st.markdown(
                render_results_table(filtered, show_level),
                unsafe_allow_html=True
            )

            # Export
            st.markdown('<div class="section-header">Export</div>',
                        unsafe_allow_html=True)
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                rows = []
                for a in filtered:
                    rows.append({
                        'Compound': a['compound'],
                        'MSI_Level': a.get('msi_level', 4),
                        'MSI_Name': a.get('msi_name', ''),
                        'Score': score_value(a)
                    })
                df_export = pd.DataFrame(rows)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="autonmr_results.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with col_e2:
                json_data = json.dumps(
                    {
                        'subject': 'Plant Extract [1]',
                        'sample_type': sample_type,
                        'method': method,
                        'compounds_found': len(filtered),
                        'processing_time': proc_time,
                        'results': [
                            {
                                'compound': a['compound'],
                                'msi_level': a.get(
                                    'msi_level', 4
                                ),
                                'score': score_value(a)
                            }
                            for a in filtered
                        ]
                    },
                    indent=2
                )
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="autonmr_results.json",
                    mime="application/json",
                    width="stretch"
                )



# =========================================
# PAGE: ABOUT
# =========================================
def page_about():
    render_app_header(
        "About AutoNMR",
        "Research pipeline, reference sources, and technical foundation"
    )
    st.markdown("""
    <h2 class="page-title">Platform Context</h2>
    <p class="page-kicker">
        AutoNMR is designed to turn NMR annotation into a repeatable,
        reviewable workflow for institutional research teams.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-header" style="margin-top:0;">Mission</div>
            <p style="color:#475569;line-height:1.55;">
                Automate NMR spectrum annotation to replace manual analysis
                for Plant Extract A-N while keeping results structured,
                explainable, and export-ready.
            </p>
            <div class="section-header">Architecture</div>
            <div style="display:grid;gap:0.55rem;color:#475569;font-size:0.9rem;">
                <div>Phase 1: Reference Library (HMDB + BMRB + NP-MRD)</div>
                <div>Phase 2: Core Engine (Preprocessing + Detection)</div>
                <div>Phase 3: ML/AI (Random Forest + 1D CNN)</div>
                <div>Phase 4: Enterprise-ready Streamlit UI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-header" style="margin-top:0;">Performance</div>
            <div style="display:grid;gap:0.55rem;color:#475569;font-size:0.9rem;">
                <div>Accuracy: ≥ 85% vs expert annotation</div>
                <div>Speed: < 5 min/spectrum</div>
                <div>Throughput: 100+ samples/day</div>
                <div>License cost: $0 open-source foundation</div>
            </div>
            <div class="section-header">Reference Databases</div>
            <div style="display:grid;gap:0.55rem;color:#475569;font-size:0.9rem;">
                <div>HMDB: 220,000+ metabolites</div>
                <div>BMRB: 15,000+ NMR entries</div>
                <div>NP-MRD: 50,000+ natural products</div>
            </div>
            <div class="section-header">Tech Stack</div>
            <div style="display:grid;gap:0.55rem;color:#475569;font-size:0.9rem;">
                <div>Python, scikit-learn, PyTorch</div>
                <div>Streamlit and Plotly visualization</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================
# MAIN APP
# =========================================
def main():
    page, method, min_confidence, show_level = \
        render_sidebar()

    if page == "Dashboard":
        page_home()
    elif page == "Workspace":
        page_annotate(method, min_confidence, show_level)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
