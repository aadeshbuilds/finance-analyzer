# =============================================================
# app/streamlit_app.py
# AI Personal Finance Analyzer — Premium Dark Dashboard
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.model_utils import (
    load_config,
    load_pipeline,
    engineer_features,
    predict_fraud
)

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="FinShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# PREMIUM CSS
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #030712 !important;
    color: #f1f5f9;
}

#MainMenu, footer, header, .stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ALWAYS VISIBLE ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    padding: 0 !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"] > div {
    width: 260px !important;
    padding: 0 !important;
}
button[kind="header"] {
    display: none !important;
}

/* ── radio nav ──────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div { flex-direction: column; gap: 4px; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.1);
    color: #a5b4fc;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border-left: 3px solid #6366f1;
}

/* ── buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.5);
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
}
.stButton > button:active { transform: translateY(0px); }

/* ── file uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.03) !important;
    border: 2px dashed rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(99,102,241,0.06) !important;
}

/* ── inputs ─────────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: #0f172a !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

/* ── dataframe ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.15);
}

/* ── spinner ─────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,102,241,0.15);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #818cf8 !important;
}

/* ── progress bar ───────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* ── hero ────────────────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #080f1e 0%, #0f172a 40%, #1a1040 100%);
    border-bottom: 1px solid rgba(99,102,241,0.15);
    padding: 3rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    animation: pulse-glow 4s ease-in-out infinite;
}
.hero-container::after {
    content: '';
    position: absolute;
    bottom: -50%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%);
    animation: pulse-glow 4s ease-in-out infinite 2s;
}
@keyframes pulse-glow {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50%       { transform: scale(1.1); opacity: 1; }
}

.gradient-text {
    background: linear-gradient(135deg, #818cf8, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── glass card ──────────────────────────────────────────── */
.glass-card {
    background: rgba(15,23,42,0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent);
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 40px rgba(99,102,241,0.05);
}

/* ── stat card ───────────────────────────────────────────── */
.stat-card {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}
.stat-card:hover::after { transform: scaleX(1); }
.stat-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin: 0.5rem 0;
}
.stat-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stat-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }

/* ── section title ───────────────────────────────────────── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
    margin-left: 0.5rem;
}

/* ── badges ──────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-low    { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-medium { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-high   { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-critical {
    background: rgba(220,38,38,0.15);
    color: #dc2626;
    border: 1px solid rgba(220,38,38,0.5);
    animation: badge-pulse 1.5s infinite;
}
@keyframes badge-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }
    50%      { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
}

/* ── feature card ────────────────────────────────────────── */
.feature-card {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    height: 100%;
}
.feature-card:hover {
    border-color: rgba(99,102,241,0.4);
    background: rgba(99,102,241,0.05);
    transform: translateY(-4px);
}
.feature-icon {
    width: 48px; height: 48px;
    background: rgba(99,102,241,0.1);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1rem;
}

/* ── step card ───────────────────────────────────────────── */
.step-card {
    background: #0f172a;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    transition: all 0.3s ease;
}
.step-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-3px);
}
.step-number {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    color: white;
    margin-bottom: 1rem;
}

/* ── alerts ──────────────────────────────────────────────── */
.alert-success {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #10b981;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-weight: 500;
}
.alert-danger {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ef4444;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-weight: 500;
}
.alert-info {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #818cf8;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-weight: 500;
}

/* ── divider ─────────────────────────────────────────────── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), rgba(139,92,246,0.4), transparent);
    margin: 2rem 0;
}

.content-wrapper { padding: 2rem 3rem; }

@keyframes logo-pulse {
    0%,100% { filter: drop-shadow(0 0 8px rgba(99,102,241,0.6)); }
    50%      { filter: drop-shadow(0 0 20px rgba(99,102,241,0.9)); }
}
.logo-icon { animation: logo-pulse 3s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# PLOTLY DARK THEME
# =============================================================
PLOTLY_THEME = dict(
    paper_bgcolor='#0f172a',
    plot_bgcolor='#0f172a',
    font=dict(color='#94a3b8', family='Inter'),
    title_font=dict(color='#e2e8f0', size=15, family='Inter'),
    xaxis=dict(
        gridcolor='rgba(99,102,241,0.08)',
        linecolor='rgba(99,102,241,0.15)',
        tickfont=dict(color='#64748b')
    ),
    yaxis=dict(
        gridcolor='rgba(99,102,241,0.08)',
        linecolor='rgba(99,102,241,0.15)',
        tickfont=dict(color='#64748b')
    ),
    legend=dict(
        bgcolor='rgba(15,23,42,0.8)',
        bordercolor='rgba(99,102,241,0.2)',
        borderwidth=1,
        font=dict(color='#94a3b8')
    ),
    colorway=['#6366f1','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ef4444']
)

COLORS = {
    'primary':  '#6366f1',
    'purple':   '#8b5cf6',
    'cyan':     '#06b6d4',
    'green':    '#10b981',
    'amber':    '#f59e0b',
    'red':      '#ef4444',
    'low':      '#10b981',
    'medium':   '#f59e0b',
    'high':     '#ef4444',
    'critical': '#dc2626',
}


# =============================================================
# CACHE
# =============================================================
@st.cache_resource
def load_model_and_config():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(base)

        possible_model_dirs = [
            os.path.join(root, "models"),
            os.path.join(base, "models"),
            "models",
        ]

        models_dir = None
        for path in possible_model_dirs:
            if os.path.exists(os.path.join(path, "model_config.json")):
                models_dir = path
                break

        if models_dir is None:
            st.error("Models directory not found!")
            return None, None, False

        config_path = os.path.join(models_dir, "model_config.json")
        config      = load_config(config_path)
        pipeline    = load_pipeline("best", models_dir)
        return config, pipeline, True

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False


# =============================================================
# HELPERS
# =============================================================
def get_badge(risk: str) -> str:
    icons   = {'LOW':'●','MEDIUM':'▲','HIGH':'⚠','CRITICAL':'🚨'}
    classes = {'LOW':'badge-low','MEDIUM':'badge-medium','HIGH':'badge-high','CRITICAL':'badge-critical'}
    return f'<span class="badge {classes.get(risk,"badge-low")}">{icons.get(risk,"●")} {risk}</span>'


def make_gauge(prob: float) -> go.Figure:
    color = (
        COLORS['green']    if prob < 0.3 else
        COLORS['amber']    if prob < 0.7 else
        COLORS['high']     if prob < 0.9 else
        COLORS['critical']
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 2),
        number={'suffix':'%','font':{'size':42,'color':color,'family':'Inter'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#334155','tickfont':{'color':'#64748b','size':11}},
            'bar':{'color':color,'thickness':0.25},
            'bgcolor':'#0f172a',
            'borderwidth':0,
            'steps':[
                {'range':[0,30],  'color':'rgba(16,185,129,0.08)'},
                {'range':[30,70], 'color':'rgba(245,158,11,0.08)'},
                {'range':[70,90], 'color':'rgba(239,68,68,0.08)'},
                {'range':[90,100],'color':'rgba(220,38,38,0.12)'},
            ],
            'threshold':{'line':{'color':'#6366f1','width':2},'thickness':0.8,'value':50}
        },
        domain={'x':[0,1],'y':[0,1]}
    ))
    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        height=260,
        margin=dict(l=30,r=30,t=10,b=10)
    )
    return fig


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    type_mapping = {'CASH_IN':0,'CASH_OUT':1,'DEBIT':2,'PAYMENT':3,'TRANSFER':4}
    if 'type' in df.columns and df['type'].dtype == object:
        df['type'] = df['type'].map(type_mapping).fillna(0)
    if 'step' not in df.columns:
        df['step'] = 1
    return df


def batch_predict(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    df_p = preprocess_df(df.copy())
    probs, risks, preds = [], [], []
    progress = st.progress(0, text="Running AI analysis...")

    for i, (_, row) in enumerate(df_p.iterrows()):
        try:
            X    = engineer_features(row.to_dict())
            prob = float(pipeline.predict_proba(X)[0][1])
            pred = int(pipeline.predict(X)[0])
        except:
            prob, pred = 0.0, 0

        risk = (
            'LOW'      if prob < 0.3 else
            'MEDIUM'   if prob < 0.7 else
            'HIGH'     if prob < 0.9 else
            'CRITICAL'
        )
        probs.append(round(prob, 4))
        risks.append(risk)
        preds.append(pred)

        if i % max(1, len(df_p) // 50) == 0:
            progress.progress(
                min(int((i+1)/len(df_p)*100), 100),
                text=f"Analyzing transaction {i+1:,} of {len(df_p):,}..."
            )

    progress.empty()
    df['fraud_probability']  = probs
    df['risk_level']         = risks
    df['is_fraud_predicted'] = preds
    return df


# =============================================================
# SIDEBAR
# =============================================================
def render_sidebar(config):
    with st.sidebar:
        st.markdown("""
        <div style='padding:2rem 1.5rem 1rem;'>
            <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem'>
                <span class='logo-icon' style='font-size:2rem'>🛡️</span>
                <div>
                    <div style='font-size:1.2rem;font-weight:800;
                                background:linear-gradient(135deg,#818cf8,#a78bfa);
                                -webkit-background-clip:text;
                                -webkit-text-fill-color:transparent;
                                background-clip:text;line-height:1.2'>
                        FinShield
                    </div>
                    <div style='font-size:0.7rem;color:#475569;font-weight:500;
                                letter-spacing:0.1em;text-transform:uppercase'>
                        AI Fraud Detection
                    </div>
                </div>
            </div>
        </div>
        <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);margin:0 1rem 1rem'></div>
        <div style='padding:0 1.2rem;font-size:0.7rem;font-weight:600;color:#334155;
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem'>
            Navigation
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🏠  Home",
             "📁  Upload & Analyze",
             "🔍  Single Transaction",
             "📊  Model Performance",
             "ℹ️  About"],
            label_visibility="collapsed"
        )

        if config:
            best    = config['best_model']
            metrics = config['models'][best]
            st.markdown(f"""
            <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);margin:1rem'></div>
            <div style='padding:0 1.2rem;font-size:0.7rem;font-weight:600;color:#334155;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem'>
                Live Model
            </div>
            """, unsafe_allow_html=True)

            for label, val in [
                ("Model",   best.replace('_',' ').title()),
                ("Version", f"v{config['model_version']}"),
                ("AUC",     f"{metrics['roc_auc']:.4f}"),
                ("F1",      f"{metrics['f1_score']:.4f}"),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:0.5rem 1.2rem;margin-bottom:2px'>
                    <span style='font-size:0.8rem;color:#475569;font-weight:500'>{label}</span>
                    <span style='font-size:0.8rem;color:#818cf8;font-weight:600;
                                 background:rgba(99,102,241,0.1);padding:2px 10px;border-radius:20px'>
                        {val}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div style='height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);margin:1rem'></div>
        <div style='padding:1rem 1.2rem;'>
            <div style='display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;color:#10b981'>
                <div style='width:8px;height:8px;background:#10b981;border-radius:50%;animation:badge-pulse 2s infinite'></div>
                System Online
            </div>
            <div style='font-size:0.72rem;color:#334155;margin-top:0.3rem'>All systems operational</div>
        </div>
        """, unsafe_allow_html=True)

    return page


# =============================================================
# PAGE: HOME  (no model metrics here)
# =============================================================
def render_home():
    st.markdown("""
    <div class='hero-container'>
        <div style='position:relative;z-index:1;max-width:700px'>
            <div style='display:inline-flex;align-items:center;gap:0.5rem;
                        background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
                        border-radius:20px;padding:0.3rem 1rem;font-size:0.8rem;color:#818cf8;
                        font-weight:500;margin-bottom:1.5rem;letter-spacing:0.02em'>
                <span style='width:6px;height:6px;background:#10b981;border-radius:50%;display:inline-block'></span>
                Powered by XGBoost • AUC 0.9998
            </div>
            <h1 style='font-size:3.5rem;font-weight:900;line-height:1.1;
                       margin-bottom:1rem;letter-spacing:-0.02em'>
                <span class='gradient-text'>AI-Powered</span><br>
                <span style='color:#e2e8f0'>Fraud Detection</span>
            </h1>
            <p style='font-size:1.1rem;color:#64748b;line-height:1.7;max-width:550px;font-weight:400'>
                Enterprise-grade anomaly detection for financial transactions.
                Upload your bank statement and get instant AI-powered fraud analysis.
            </p>
            <div style='display:flex;gap:1.5rem;margin-top:2rem;flex-wrap:wrap'>
                <div style='display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:#64748b'>
                    <span style='color:#10b981;font-size:1rem'>✓</span> Real-time Detection
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:#64748b'>
                    <span style='color:#10b981;font-size:1rem'>✓</span> 27 ML Features
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:#64748b'>
                    <span style='color:#10b981;font-size:1rem'>✓</span> 99.98% Accuracy
                </div>
                <div style='display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:#64748b'>
                    <span style='color:#10b981;font-size:1rem'>✓</span> Instant Reports
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)

    # ── Core Features ─────────────────────────────────────────
    st.markdown("<div class='section-title'>✨ Core Features</div>",
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    features = [
        ("🛡️","Fraud Detection",
         "XGBoost + Random Forest ensemble detects fraudulent transactions in milliseconds",
         "#6366f1"),
        ("📊","Visual Analytics",
         "Interactive Plotly dashboards with real-time charts and spending breakdowns",
         "#8b5cf6"),
        ("⚡","Instant Analysis",
         "Upload any CSV and get complete fraud analysis with risk scores instantly",
         "#06b6d4"),
        ("📄","Smart Reports",
         "Download detailed CSV reports with fraud probabilities and risk levels",
         "#10b981"),
    ]
    for col,(icon,title,desc,color) in zip([c1,c2,c3,c4], features):
        with col:
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-icon' style='background:rgba({r},{g},{b},0.12)'>{icon}</div>
                <div style='font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem'>{title}</div>
                <div style='font-size:0.82rem;color:#475569;line-height:1.6'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ──────────────────────────────────────────
    st.markdown("<div class='section-title'>🚀 How It Works</div>",
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    steps = [
        ("1","Upload CSV",
         "Drop your bank statement CSV with transaction data. We support any standard format.","📁"),
        ("2","AI Analysis",
         "Our ML pipeline engineers 27 features and runs XGBoost inference on every transaction.","🤖"),
        ("3","Get Results",
         "View fraud probabilities, risk levels, interactive charts, and download your report.","📊"),
    ]
    for col,(num,title,desc,icon) in zip([c1,c2,c3], steps):
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-number'>{num}</div>
                <div style='font-size:1.5rem;margin-bottom:0.5rem'>{icon}</div>
                <div style='font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem'>{title}</div>
                <div style='font-size:0.82rem;color:#475569;line-height:1.6'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────
    st.markdown("<div class='section-title'>🔧 Tech Stack</div>",
                unsafe_allow_html=True)

    techs = [
        ("🐍","Python 3.14"),("🤖","XGBoost"),("🌲","Scikit-learn"),
        ("📊","Streamlit"),("📈","Plotly"),("🐼","Pandas"),
        ("🔢","NumPy"),("💾","Joblib"),
    ]
    tech_html = "<div style='display:flex;flex-wrap:wrap;gap:0.8rem;margin-bottom:2rem'>"
    for icon,name in techs:
        tech_html += f"""
        <div style='display:inline-flex;align-items:center;gap:0.4rem;
                    background:#0f172a;border:1px solid rgba(99,102,241,0.15);
                    border-radius:8px;padding:0.4rem 0.8rem;
                    font-size:0.82rem;color:#94a3b8;font-weight:500'>
            {icon} {name}
        </div>"""
    st.markdown(tech_html + "</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# PAGE: UPLOAD & ANALYZE
# =============================================================
def render_upload(pipeline, config):
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📁 Upload & Analyze</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='alert-info' style='margin-bottom:1.5rem'>
        <span>ℹ️</span>
        <span>Required columns:
            <code style='background:rgba(99,102,241,0.15);padding:1px 6px;border-radius:4px'>
                step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
            </code>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # generate sample CSV
    sample_data = pd.DataFrame({
        'step':           [1, 2, 3, 4, 5],
        'type':           ['PAYMENT','TRANSFER','CASH_OUT','PAYMENT','TRANSFER'],
        'amount':         [9839.64, 1864.28, 181.00, 11668.14, 7817.71],
        'oldbalanceOrg':  [170136.0, 21249.0, 181.0, 41554.0, 53860.0],
        'newbalanceOrig': [160296.36, 19384.72, 0.0, 29885.86, 46042.29],
        'oldbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
        'newbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
    })

    col_download, col_info = st.columns([1, 3])
    with col_download:
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_transactions.csv",
            mime="text/csv",
            help="Download this sample file, then upload it below to test the analyzer"
        )
    with col_info:
        st.markdown("""
        <div class='alert-info' style='margin:0'>
            <span>💡</span>
            <span>New here? Download the sample CSV above,
            then upload it below to see the analyzer in action instantly.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your CSV file here or click to browse",
        type=['csv']
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        required = ['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
        missing  = [c for c in required if c not in df.columns]

        if missing:
            st.markdown(f"<div class='alert-danger'>⚠️ Missing columns: {missing}</div>",
                        unsafe_allow_html=True)
            return

        st.markdown(f"""
        <div class='alert-success' style='margin:1rem 0'>
            ✅ Successfully loaded <strong>{len(df):,} transactions</strong> — Running AI analysis...
        </div>
        """, unsafe_allow_html=True)

        df_results = batch_predict(df.copy(), pipeline)

        # KPIs
        st.markdown("<div class='section-title'>📊 Analysis Results</div>",
                    unsafe_allow_html=True)

        total    = len(df_results)
        flagged  = int(df_results['is_fraud_predicted'].sum())
        critical = int((df_results['risk_level']=='CRITICAL').sum())
        high     = int((df_results['risk_level']=='HIGH').sum())
        safe     = total - flagged
        avg_prob = df_results['fraud_probability'].mean()

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        for col,(val,label,icon,style) in zip(
            [c1,c2,c3,c4,c5,c6],[
                (f"{total:,}",   "Total",    "💳",""),
                (f"{safe:,}",    "Safe",     "✅","color:#10b981"),
                (f"{flagged:,}", "Flagged",  "⚠️","color:#f59e0b"),
                (f"{high:,}",    "High Risk","🔴","color:#ef4444"),
                (f"{critical:,}","Critical", "🚨","color:#dc2626"),
                (f"{avg_prob:.1%}","Avg Risk","📈",""),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-icon'>{icon}</div>
                    <div class='stat-value' style='font-size:1.6rem;{style}'>{val}</div>
                    <div class='stat-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row 1
        c1,c2 = st.columns(2)
        with c1:
            rc = df_results['risk_level'].value_counts().reset_index()
            rc.columns = ['Risk Level','Count']
            fig = px.pie(rc,values='Count',names='Risk Level',
                         title='Risk Level Distribution',
                         color='Risk Level',
                         color_discrete_map={'LOW':COLORS['low'],'MEDIUM':COLORS['medium'],
                                             'HIGH':COLORS['high'],'CRITICAL':COLORS['critical']},
                         hole=0.55)
            fig.update_traces(textposition='outside',textfont_size=11)
            fig.update_layout(**PLOTLY_THEME,height=380,showlegend=True,
                              annotations=[dict(text=f'<b>{total:,}</b><br>Total',
                                                x=0.5,y=0.5,font_size=14,
                                                font_color='#e2e8f0',showarrow=False)])
            st.plotly_chart(fig,use_container_width=True)

        with c2:
            fig = px.histogram(df_results,x='fraud_probability',nbins=40,
                               title='Fraud Probability Distribution',
                               color_discrete_sequence=[COLORS['primary']])
            fig.update_layout(**PLOTLY_THEME,height=380,
                              xaxis_title='Fraud Probability',
                              yaxis_title='Transaction Count',bargap=0.05)
            fig.add_vline(x=0.5,line_dash='dash',line_color=COLORS['red'],
                          line_width=2,annotation_text='<b>Decision Threshold</b>',
                          annotation_font_color=COLORS['red'],annotation_font_size=11)
            fig.update_traces(marker_line_width=0,opacity=0.85)
            st.plotly_chart(fig,use_container_width=True)

        # Charts row 2
        c1,c2 = st.columns(2)
        with c1:
            fig = px.box(df_results,x='risk_level',y='amount',color='risk_level',
                         title='Transaction Amount by Risk Level',
                         color_discrete_map={'LOW':COLORS['low'],'MEDIUM':COLORS['medium'],
                                             'HIGH':COLORS['high'],'CRITICAL':COLORS['critical']},
                         category_orders={'risk_level':['LOW','MEDIUM','HIGH','CRITICAL']})
            fig.update_layout(**PLOTLY_THEME,height=380,showlegend=False,
                              xaxis_title='Risk Level',yaxis_title='Amount ($)')
            st.plotly_chart(fig,use_container_width=True)

        with c2:
            if 'type' in df_results.columns:
                type_map = {0:'CASH_IN',1:'CASH_OUT',2:'DEBIT',3:'PAYMENT',4:'TRANSFER'}
                df_results['type_name'] = df_results['type'].map(type_map)
                tf = df_results.groupby('type_name').agg(
                    fraud=('is_fraud_predicted','sum'),
                    total=('is_fraud_predicted','count')
                ).reset_index()
                tf['fraud_rate'] = (tf['fraud']/tf['total']*100).round(2)
                fig = px.bar(tf,x='type_name',y='fraud_rate',
                             title='Fraud Rate by Transaction Type',
                             color='fraud_rate',
                             color_continuous_scale=[[0,COLORS['low']],[0.5,COLORS['medium']],[1,COLORS['red']]],
                             text=tf['fraud_rate'].apply(lambda x: f'{x:.1f}%'))
                fig.update_layout(**PLOTLY_THEME,height=380,showlegend=False,
                                  xaxis_title='Transaction Type',yaxis_title='Fraud Rate (%)',
                                  coloraxis_showscale=False)
                fig.update_traces(textposition='outside',marker_line_width=0)
                st.plotly_chart(fig,use_container_width=True)

        # Table
        st.markdown("<div class='section-title'>🔍 Transaction Details</div>",
                    unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            risk_filter = st.multiselect("Risk Level",
                ['LOW','MEDIUM','HIGH','CRITICAL'],
                default=['HIGH','CRITICAL'])
        with c2:
            min_prob = st.slider("Min Probability",0.0,1.0,0.5,0.05)
        with c3:
            show_all = st.checkbox("Show All Transactions")

        df_display = (df_results if show_all else
                      df_results[df_results['risk_level'].isin(risk_filter) &
                                 (df_results['fraud_probability'] >= min_prob)])

        cols = [c for c in ['step','type','amount','oldbalanceOrg','newbalanceOrig',
                             'fraud_probability','risk_level','is_fraud_predicted']
                if c in df_display.columns]

        st.dataframe(
            df_display[cols].style.background_gradient(
                subset=['fraud_probability'],cmap='RdYlGn_r'
            ).format({'amount':'${:,.2f}','oldbalanceOrg':'${:,.2f}',
                      'newbalanceOrig':'${:,.2f}','fraud_probability':'{:.4f}'}),
            use_container_width=True, height=420
        )

        st.markdown("<br>", unsafe_allow_html=True)
        _,c2,_ = st.columns([1,1,1])
        with c2:
            st.download_button(
                "⬇️  Download Full Report (CSV)",
                data=df_results.to_csv(index=False),
                file_name="finshield_fraud_report.csv",
                mime="text/csv"
            )

        st.session_state['df_results']    = df_results
        st.session_state['analysis_done'] = True

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# PAGE: SINGLE TRANSACTION
# =============================================================
def render_single(pipeline):
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Single Transaction Analyzer</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='alert-info' style='margin-bottom:1.5rem'>
        <span>⚡</span>
        <span>Enter any transaction details below for instant AI fraud scoring</span>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        tx_type  = st.selectbox("Transaction Type",['PAYMENT','TRANSFER','CASH_OUT','CASH_IN','DEBIT'])
        amount   = st.number_input("Amount ($)",0.0,10_000_000.0,1000.0,100.0,format="%.2f")
        step     = st.number_input("Time Step (Hour)",1,744,100)
    with c2:
        old_orig = st.number_input("Sender Balance Before ($)",0.0,1e8,10000.0,100.0,format="%.2f")
        new_orig = st.number_input("Sender Balance After ($)",0.0,1e8,9000.0,100.0,format="%.2f")
        old_dest = st.number_input("Receiver Balance Before ($)",0.0,1e8,0.0,100.0,format="%.2f")
        new_dest = st.number_input("Receiver Balance After ($)",0.0,1e8,1000.0,100.0,format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    _,c2,_ = st.columns([1,1,1])
    with c2:
        analyze = st.button("🔍  Analyze Transaction")

    if analyze:
        type_map = {'CASH_IN':0,'CASH_OUT':1,'DEBIT':2,'PAYMENT':3,'TRANSFER':4}
        tx = {
            'step':step,'type':type_map[tx_type],'amount':amount,
            'oldbalanceOrg':old_orig,'newbalanceOrig':new_orig,
            'oldbalanceDest':old_dest,'newbalanceDest':new_dest,
        }
        with st.spinner("Running AI inference..."):
            result = predict_fraud(tx,model_name="best",models_dir="models/")
            time.sleep(0.3)

        st.markdown("<div class='fancy-divider'></div><div class='section-title'>🎯 Analysis Result</div>",
                    unsafe_allow_html=True)

        c1,c2,c3 = st.columns([1.2,1,1])
        with c1:
            st.plotly_chart(make_gauge(result['fraud_probability']),use_container_width=True)
        with c2:
            verdict = "⚠️ FRAUD DETECTED" if result['is_fraud'] else "✅ SAFE"
            vcolor  = "#ef4444" if result['is_fraud'] else "#10b981"
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;padding:2rem'>
                <div style='font-size:2rem;margin-bottom:1rem'>{'🚨' if result['is_fraud'] else '🛡️'}</div>
                <div style='font-size:1.3rem;font-weight:800;color:{vcolor};margin-bottom:1rem'>{verdict}</div>
                <div style='margin:1rem 0'>{get_badge(result['risk_level'])}</div>
                <div style='font-size:0.85rem;color:#475569;margin-top:1rem'>
                    Fraud Probability
                    <div style='font-size:1.8rem;font-weight:800;color:{vcolor};margin-top:0.3rem'>
                        {result['fraud_pct']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            rows_html = ""
            for label,val in [("Type",tx_type),("Amount",f"${amount:,.2f}"),
                               ("Step",str(step)),("Balance Δ",f"-${amount:,.2f}"),
                               ("Risk",result['risk_level'])]:
                rows_html += f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:0.7rem 0;border-bottom:1px solid rgba(99,102,241,0.08)'>
                    <span style='font-size:0.82rem;color:#475569;font-weight:500'>{label}</span>
                    <span style='font-size:0.85rem;color:#e2e8f0;font-weight:600'>{val}</span>
                </div>"""
            st.markdown("**Summary**")
            for label,val in [
                ("Type",     tx_type),
                ("Amount",   f"${amount:,.2f}"),
                ("Step",     str(step)),
                ("Balance Δ",f"-${amount:,.2f}"),
                ("Risk",     result['risk_level']),
            ]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;
                            padding:0.5rem 0;
                            border-bottom:1px solid rgba(99,102,241,0.08)'>
                    <span style='color:#475569;font-size:0.85rem'>{label}</span>
                    <span style='color:#e2e8f0;font-weight:600;font-size:0.85rem'>{val}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# PAGE: MODEL PERFORMANCE
# =============================================================
def render_performance(config):
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(245,158,11,0.08);
                border:1px solid rgba(245,158,11,0.25);
                border-radius:12px;padding:1rem 1.2rem;
                color:#f59e0b;font-size:0.85rem;
                margin-bottom:1rem'>
        ⚠️ <strong>Disclaimer:</strong> FinShield AI is a demonstration project 
        built for educational purposes. Results should not be used as the sole 
        basis for fraud detection decisions. Always consult your bank or financial 
        institution for official fraud reporting.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Model Performance</div>",
                unsafe_allow_html=True)

    model_icons = {
        'xgboost':('🚀',COLORS['primary']),
        'random_forest':('🌲',COLORS['green']),
        'logistic_regression':('📈',COLORS['cyan']),
    }

    model_items = list(config['models'].items())
    c1,c2,c3 = st.columns(3)

    for col, (name, metrics) in zip([c1,c2,c3], model_items):
        icon,color = model_icons.get(name,('🤖','#6366f1'))
        is_best = name == config['best_model']
        with col:
            if is_best:
                st.success(f"{icon} {name.replace('_',' ').title()} ⭐ BEST")
            else:
                st.info(f"{icon} {name.replace('_',' ').title()}")

            st.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")
            st.metric("F1 Score",  f"{metrics['f1_score']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("Recall",    f"{metrics['recall']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Comparison Chart</div>",
                unsafe_allow_html=True)

    df_m = pd.DataFrame([{'Model':n.replace('_',' ').title(),**m}
                          for n,m in config['models'].items()])
    fig  = go.Figure()
    for metric,color,label in [
        ('roc_auc',COLORS['primary'],'ROC-AUC'),
        ('f1_score',COLORS['green'],'F1 Score'),
        ('precision',COLORS['cyan'],'Precision'),
        ('recall',COLORS['amber'],'Recall'),
    ]:
        fig.add_trace(go.Bar(
            name=label,x=df_m['Model'],y=df_m[metric],
            marker_color=color,marker_line_width=0,opacity=0.85,
            text=df_m[metric].apply(lambda x:f'{x:.4f}'),
            textposition='outside',textfont=dict(size=10,color='#94a3b8')
        ))
    fig.update_layout(
        title='Model Comparison',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(color='#94a3b8',family='Inter'),
        title_font=dict(color='#e2e8f0',size=15,family='Inter'),
        legend=dict(
            bgcolor='rgba(15,23,42,0.8)',
            bordercolor='rgba(99,102,241,0.2)',
            borderwidth=1,
            font=dict(color='#94a3b8')
        ),
        barmode='group',
        height=420,
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.08)',
            linecolor='rgba(99,102,241,0.15)',
            tickfont=dict(color='#64748b')
        ),
        yaxis=dict(
            range=[0.88,1.025],
            gridcolor='rgba(99,102,241,0.08)',
            linecolor='rgba(99,102,241,0.15)',
            tickfont=dict(color='#64748b')
        ),
        bargap=0.2,
        bargroupgap=0.05
    )
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("<div class='section-title'>🖼️ Evaluation Charts</div>",
                unsafe_allow_html=True)

    tabs = st.tabs(["ROC Curves","Precision-Recall","Confusion Matrices",
                    "Feature Importance","Model Comparison"])
    chart_files = [
        'reports/06_roc_curves.png',
        'reports/07_precision_recall_curves.png',
        'reports/08_confusion_matrices.png',
        'reports/09_feature_importance.png',
        'reports/10_model_comparison.png',
    ]
    for tab,filepath in zip(tabs,chart_files):
        with tab:
            if os.path.exists(filepath):
                st.image(filepath,use_column_width=True)
            else:
                st.markdown("<div class='alert-info'>Run <code>python src/evaluate.py</code> to generate charts</div>",
                            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# PAGE: ABOUT  (model metrics moved here)
# =============================================================
def render_about():
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>ℹ️ About FinShield AI</div>",
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:1.1rem;font-weight:700;color:#818cf8;margin-bottom:1rem'>
                About This Project
            </div>
            <p style='color:#64748b;line-height:1.8;font-size:0.9rem'>
                FinShield AI is a production-grade machine learning system for detecting
                fraudulent financial transactions. Built from scratch using Python,
                Scikit-learn, XGBoost, and Streamlit.
            </p>
            <p style='color:#64748b;line-height:1.8;font-size:0.9rem;margin-top:0.8rem'>
                Trained on 58,000+ transactions with 27 engineered features achieving
                99.98% ROC-AUC score — comparable to enterprise fraud detection systems.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-size:1.1rem;font-weight:700;color:#818cf8;margin-bottom:1rem'>
                ML Pipeline
            </div>
        """, unsafe_allow_html=True)
        for icon,step in [
            ("📥","Raw CSV Upload"),
            ("🧹","Preprocessing Pipeline"),
            ("⚙️","Feature Engineering (27 features)"),
            ("🤖","XGBoost Inference"),
            ("📊","Risk Scoring & Visualization"),
            ("📄","Report Generation"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;padding:0.5rem 0;
                        border-bottom:1px solid rgba(99,102,241,0.08)'>
                <span style='font-size:1rem'>{icon}</span>
                <span style='font-size:0.85rem;color:#94a3b8;font-weight:500'>{step}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # model metrics here
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Model Performance Metrics</div>",
                unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(icon,val,label) in zip([c1,c2,c3,c4,c5],[
        ("🎯","99.98%","ROC-AUC Score"),
        ("⚡","99.91%","F1 Score"),
        ("🎯","100%","Precision"),
        ("🔍","99.82%","Recall"),
        ("💾","58,213","Training Size"),
    ]):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>{icon}</div>
                <div class='stat-value'>{val}</div>
                <div class='stat-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # tech stack
    st.markdown("<div class='section-title'>🔧 Tech Stack</div>",
                unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    techs = [
        ("🐍","Python 3.14","Core language","#6366f1"),
        ("🤖","XGBoost","Primary ML model","#8b5cf6"),
        ("🌲","Scikit-learn","ML pipeline","#06b6d4"),
        ("📊","Streamlit","Web framework","#10b981"),
        ("📈","Plotly","Interactive charts","#f59e0b"),
        ("🐼","Pandas","Data processing","#ef4444"),
        ("🔢","NumPy","Numerical computing","#6366f1"),
        ("💾","Joblib","Model serialization","#8b5cf6"),
    ]
    for i,(icon,name,desc,color) in enumerate(techs):
        col = [c1,c2,c3,c4][i%4]
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        with col:
            st.markdown(f"""
            <div class='feature-card' style='margin-bottom:1rem'>
                <div class='feature-icon'
                     style='background:rgba({r},{g},{b},0.12);width:40px;height:40px;font-size:1.2rem'>
                    {icon}
                </div>
                <div style='font-size:0.9rem;font-weight:700;color:#e2e8f0;margin-bottom:0.2rem'>{name}</div>
                <div style='font-size:0.78rem;color:#475569'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# MAIN
# =============================================================
def main():
    config, pipeline, loaded = load_model_and_config()

    if not loaded:
        st.markdown("""
        <div class='content-wrapper'>
            <div class='alert-danger'>
                ⚠️ Models not found. Run <code>python src/train.py</code> first.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    page = render_sidebar(config)

    if   "Home"        in page: render_home()
    elif "Upload"      in page: render_upload(pipeline, config)
    elif "Single"      in page: render_single(pipeline)
    elif "Performance" in page: render_performance(config)
    elif "About"       in page: render_about()


if __name__ == "__main__":
    main()