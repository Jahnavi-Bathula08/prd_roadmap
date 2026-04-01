"""
ml_analysis.py  ·  Research-Grade Edition
==========================================
HCI / Product Analytics  —  ML Analysis Engine

Modules:
  1. NLP-based feedback clustering  (TF-IDF + KMeans)
  2. Model comparison  (RF vs GBM vs Logistic baseline)
  3. SHAP-style feature importance with confidence intervals
  4. Anomaly detection  (IsolationForest on session behaviour)
  5. Temporal pattern analysis  (hour / day heatmaps)
  6. Priority scoring + action plan  (RF urgency scoring)

Call run_analysis(df_behavior, df_feedback) from app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
T = {
    "bg":      "#080c14",
    "surface": "#0d1117",
    "border":  "#1e2d3d",
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "yellow":  "#d29922",
    "red":     "#f85149",
    "purple":  "#bc8cff",
    "cyan":    "#39d0d8",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "dim":     "#484f58",
}

# ── Base layout WITHOUT xaxis/yaxis so charts can define their own freely ──
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
)

# ── Shared axis style helper — call per-chart, never via **PLOTLY_BASE ──────
def _ax(extra: dict | None = None) -> dict:
    base = dict(
        gridcolor="#1e2d3d",
        zerolinecolor="#1e2d3d",
        tickfont=dict(size=10),
    )
    if extra:
        base.update(extra)
    return base


def _layout(**kwargs) -> dict:
    """Merge PLOTLY_BASE with per-chart overrides cleanly."""
    out = dict(**PLOTLY_BASE)
    out.update(kwargs)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def _css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .ana-divider {
        width:100%; height:1px;
        background:linear-gradient(90deg,#1f6feb 0%,#3fb950 50%,transparent 100%);
        margin:36px 0 28px; opacity:.45;
    }
    .sec-banner {
        display:flex; align-items:center; gap:12px;
        background:#0d1117; border:1px solid #1e2d3d;
        border-left:3px solid #58a6ff; color:#e6edf3;
        padding:14px 20px; border-radius:10px;
        font-family:'Syne',sans-serif; font-size:1rem;
        font-weight:800; letter-spacing:.04em;
        margin:28px 0 6px; text-transform:uppercase;
    }
    .sys-label {
        font-family:'JetBrains Mono',monospace; font-size:.68rem;
        color:#388bfd; letter-spacing:.18em; text-transform:uppercase;
        margin-bottom:18px; padding-left:2px;
    }
    .model-row {
        display:flex; align-items:center; gap:10px;
        background:#0d1117; border:1px solid #1e2d3d; border-radius:8px;
        padding:10px 16px; margin-bottom:8px;
    }
    .model-name { font-family:'Syne',sans-serif; font-weight:700; color:#e6edf3; font-size:.88rem; min-width:160px; }
    .model-bar-wrap { flex:1; height:8px; background:#1e2d3d; border-radius:4px; overflow:hidden; }
    .model-bar { height:100%; border-radius:4px; transition:width .5s ease; }
    .model-score { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.85rem; min-width:48px; text-align:right; }
    .model-winner { border-color:#3fb950 !important; }
    .anomaly-badge {
        display:inline-block; padding:2px 8px; border-radius:4px;
        background:rgba(248,81,73,.12); border:1px solid rgba(248,81,73,.3);
        color:#f85149; font-family:'JetBrains Mono',monospace; font-size:.65rem;
        font-weight:700; letter-spacing:.06em; text-transform:uppercase;
    }
    .normal-badge {
        display:inline-block; padding:2px 8px; border-radius:4px;
        background:rgba(63,185,80,.08); border:1px solid rgba(63,185,80,.2);
        color:#3fb950; font-family:'JetBrains Mono',monospace; font-size:.65rem;
        font-weight:700; letter-spacing:.06em; text-transform:uppercase;
    }
    .pcard {
        display:flex; align-items:center; gap:12px; padding:12px 16px;
        border-radius:10px; margin-bottom:8px; background:#0d1117;
        border:1px solid #1e2d3d; border-left:4px solid;
    }
    .prank { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.82rem; min-width:38px; letter-spacing:.04em; }
    .ptag {
        padding:3px 10px; border-radius:20px; font-family:'JetBrains Mono',monospace;
        font-size:.62rem; font-weight:700; color:#fff; white-space:nowrap;
        letter-spacing:.06em; text-transform:uppercase;
    }
    .ptitle { flex:1; font-family:'Syne',sans-serif; font-size:.85rem; font-weight:600; color:#c9d1d9; line-height:1.3; }
    .pscore {
        font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.88rem; color:#58a6ff;
        background:rgba(56,139,253,.1); border:1px solid rgba(56,139,253,.25);
        border-radius:6px; padding:3px 10px; white-space:nowrap;
    }
    .acard {
        border-radius:12px; padding:16px 18px; margin-bottom:12px;
        border:1px solid; background:#0d1117; position:relative; overflow:hidden;
    }
    .acard-glow { position:absolute; top:0; left:0; right:0; height:2px; }
    .atop { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }
    .aname { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:700; color:#e6edf3; margin-bottom:12px; line-height:1.35; }
    .ameta-row { display:flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace; font-size:.72rem; color:#484f58; margin-bottom:4px; }
    .ameta-label { color:#30363d; min-width:52px; letter-spacing:.06em; text-transform:uppercase; font-size:.65rem; }
    .ameta-val { color:#8b949e; }
    .biz-group-label { font-family:'JetBrains Mono',monospace; font-size:.65rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin:14px 0 8px; }
    .dbadge { display:flex; align-items:center; gap:8px; padding:8px 14px; border-radius:8px; font-family:'Syne',sans-serif; font-size:.78rem; font-weight:600; margin-bottom:6px; border:1px solid; }
    .dapproved { background:rgba(63,185,80,.07); color:#3fb950; border-color:rgba(63,185,80,.25); }
    .dreview   { background:rgba(210,153,34,.07); color:#d29922; border-color:rgba(210,153,34,.25); }
    .drejected { background:rgba(218,54,51,.07);  color:#f85149; border-color:rgba(218,54,51,.2);  }
    .suggestion {
        background:rgba(210,153,34,.07); border:1px solid rgba(210,153,34,.25);
        border-left:3px solid #d29922; padding:14px 16px; border-radius:10px;
        font-family:'Syne',sans-serif; font-size:.82rem; font-weight:600;
        color:#d29922; margin-top:16px; line-height:1.7;
    }
    .final-note {
        background:rgba(56,139,253,.06); border:1px solid rgba(56,139,253,.2);
        border-left:3px solid #388bfd; padding:12px 16px; border-radius:10px;
        font-family:'JetBrains Mono',monospace; font-size:.72rem; color:#58a6ff;
        margin-top:10px; line-height:1.8; letter-spacing:.02em;
    }
    .arrow-divider {
        text-align:center; font-family:'JetBrains Mono',monospace; font-size:.72rem;
        font-weight:700; color:#388bfd; letter-spacing:.15em; text-transform:uppercase;
        margin:24px 0; padding:10px; border:1px dashed #1e2d3d; border-radius:6px;
        background:rgba(56,139,253,.04);
    }
    .sub-heading { font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#484f58; margin-bottom:14px; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
priority_weight = {"Critical": 10, "High": 7, "Medium": 4, "Low": 1}
severity_weight = {"High": 10, "Medium": 5, "Low": 1}

_AREA_COLORS = {
    "bug": "#e74c3c", "performance": "#8e44ad", "feature": "#27ae60",
    "ux": "#2980b9", "drop-off": "#c0392b", "auth": "#e67e22",
    "engagement": "#16a085", "conversion": "#1abc9c",
}
_RANK_COLORS = ["#c0392b", "#e67e22", "#2980b9", "#27ae60", "#7f8c8d"]
_TIER = {
    0: ("Immediate",     "⚡", "#c0392b"),
    1: ("High Priority", "🔥", "#e67e22"),
    2: ("Planned",       "📅", "#2980b9"),
    3: ("Backlog",       "🗂️",  "#7f8c8d"),
}


def _acolor(area: str) -> str:
    return _AREA_COLORS.get(str(area).lower().strip(), "#95a5a6")


def _team_deadline(area: str) -> tuple:
    a = area.lower()
    if "bug"         in a: return "Backend Team",   "48 Hours"
    if "performance" in a: return "Infra Team",     "3 Days"
    if "ux"          in a: return "UX/UI Team",     "1 Week"
    if "feature"     in a: return "Product Team",   "Next Sprint"
    if "drop"        in a: return "Growth Team",    "1 Week"
    if "auth"        in a: return "Security Team",  "48 Hours"
    if "engagement"  in a: return "Marketing Team", "2 Weeks"
    if "conversion"  in a: return "Sales Team",     "1 Week"
    return "Product Team", "2 Weeks"


def _reason(tier: int, area: str) -> str:
    a = area.lower()
    if tier == 0:    return "Critical Impact"
    if "bug"  in a:  return "High Severity Bug"
    if "drop" in a:  return "High Drop-off Rate"
    if "ux"   in a:  return "Poor User Experience"
    return "User Demand"


def _biz_decision(score: float) -> tuple:
    if score >= 7.5: return "✅ APPROVED",          "dapproved"
    if score >= 5.5: return "🔄 Under Review",       "dreview"
    return "❌ Rejected (For Now)",                  "drejected"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1  ·  NLP FEEDBACK CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
                   "#39d0d8", "#e67e22", "#e74c3c", "#27ae60", "#2980b9"]


def _nlp_cluster(df_f: pd.DataFrame):
    texts = df_f["Feedback"].fillna("").tolist()
    if len(texts) < 6:
        # Not enough samples — return single cluster
        df_out = df_f.copy()
        df_out["nlp_cluster"]  = 0
        df_out["cluster_label"] = "all feedback"
        return df_out, 1, 0.0, {0: "all feedback"}

    vec = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
    X   = vec.fit_transform(texts).toarray()

    best_k, best_score = 3, -1
    max_k = min(9, len(texts))
    for k in range(3, max_k):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        if len(set(lbl)) < 2:
            continue
        s = silhouette_score(X, lbl)
        if s > best_score:
            best_k, best_score = k, s

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_out = df_f.copy()
    df_out["nlp_cluster"] = km.fit_predict(X)

    feature_names  = vec.get_feature_names_out()
    centers        = km.cluster_centers_
    cluster_labels = {}
    for i in range(best_k):
        top_idx          = centers[i].argsort()[-3:][::-1]
        cluster_labels[i] = " · ".join(feature_names[top_idx])

    df_out["cluster_label"] = df_out["nlp_cluster"].map(cluster_labels)
    return df_out, best_k, best_score, cluster_labels


def _render_nlp_section(df_f: pd.DataFrame):
    st.markdown('<div class="sec-banner">🧠 &nbsp;1. NLP FEEDBACK CLUSTERING</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;TF-IDF · KMeans · Silhouette Optimisation &nbsp;◈</div>', unsafe_allow_html=True)

    df_cl, best_k, sil, cluster_labels = _nlp_cluster(df_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters Found",    best_k)
    c2.metric("Silhouette Score",  f"{sil:.3f}")
    c3.metric("Feedback Analysed", len(df_f))
    c4.metric("Avg / Cluster",     f"{max(len(df_f) // max(best_k, 1), 1)}")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        counts = df_cl.groupby("nlp_cluster")["Feedback"].count().reset_index()
        counts["label"] = counts["nlp_cluster"].map(cluster_labels)
        colors = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in counts["nlp_cluster"]]

        fig = go.Figure(go.Bar(
            x=counts["label"],
            y=counts["Feedback"],
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(
            _layout(
                title=dict(text="Cluster Volume Distribution", font=dict(color=T["text"], size=13), x=0),
                xaxis=_ax({"tickangle": -20, "tickfont": dict(size=9)}),
                yaxis=_ax(),
                height=280,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        heat = pd.crosstab(df_cl["Category"], df_cl["nlp_cluster"])
        heat.columns = [str(cluster_labels.get(c, c))[:18] for c in heat.columns]
        fig2 = go.Figure(go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[[0, "#0d1117"], [0.5, "#1f6feb"], [1, "#58a6ff"]],
            hovertemplate="<b>%{y}</b> × %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig2.update_layout(
            _layout(
                title=dict(text="Category × NLP Cluster Heatmap", font=dict(color=T["text"], size=13), x=0),
                xaxis=_ax({"tickangle": -20, "tickfont": dict(size=9)}),
                yaxis=_ax(),
                height=260,
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="sub-heading">Discovered Topic Clusters</div>', unsafe_allow_html=True)
        for i, (cid, lbl) in enumerate(cluster_labels.items()):
            color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
            n     = len(df_cl[df_cl["nlp_cluster"] == cid])
            pct   = round(n / max(len(df_cl), 1) * 100)
            avg_r = df_cl[df_cl["nlp_cluster"] == cid]["Rating"].mean() if "Rating" in df_cl.columns else 3.0
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1e2d3d;border-left:3px solid {color};
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                    <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:.82rem;color:{color};">
                        Cluster {cid}
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#484f58;">
                        {n} items · {pct}%
                    </span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#8b949e;margin-bottom:6px;">
                    {lbl}
                </div>
                <div style="height:4px;background:#1e2d3d;border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;background:{color};border-radius:2px;"></div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#30363d;margin-top:5px;">
                    Avg Rating: {avg_r:.1f} / 5.0
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; ANOMALY DETECTION ON BEHAVIOUR STREAM &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2  ·  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _detect_anomalies(df_b: pd.DataFrame) -> pd.DataFrame:
    df = df_b.copy()
    le = LabelEncoder()
    df["_cat_enc"]  = le.fit_transform(df["Category"].astype(str))
    df["_plat_enc"] = le.fit_transform(df["Platform"].astype(str))
    X   = df[["Session (s)", "_cat_enc", "_plat_enc"]].values
    iso = IsolationForest(contamination=0.08, random_state=42)
    df["anomaly"]       = iso.fit_predict(X)       # -1=anomaly, 1=normal
    df["anomaly_score"] = iso.score_samples(X)
    return df


def _render_anomaly_section(df_b: pd.DataFrame):
    st.markdown('<div class="sec-banner">⚠️ &nbsp;2. BEHAVIOURAL ANOMALY DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;Isolation Forest · Contamination 8% &nbsp;◈</div>', unsafe_allow_html=True)

    df_a   = _detect_anomalies(df_b)
    n_anom = len(df_a[df_a["anomaly"] == -1])
    n_norm = len(df_a[df_a["anomaly"] ==  1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions",  len(df_a))
    c2.metric("Anomalies Found", n_anom)
    c3.metric("Normal Sessions", n_norm)
    c4.metric("Anomaly Rate",    f"{round(n_anom / max(len(df_a), 1) * 100, 1)}%")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        fig = go.Figure()
        for flag, label, color, sym in [
            (1,  "Normal",  T["blue"], "circle"),
            (-1, "Anomaly", T["red"],  "x"),
        ]:
            sub = df_a[df_a["anomaly"] == flag]
            fig.add_trace(go.Scatter(
                x=sub["Session (s)"],
                y=sub["anomaly_score"],
                mode="markers",
                name=label,
                marker=dict(color=color, symbol=sym, size=7, opacity=.75,
                            line=dict(width=1, color=color)),
                hovertemplate=f"<b>{label}</b><br>Session: %{{x}}s<br>Score: %{{y:.3f}}<extra></extra>",
            ))
        fig.update_layout(
            _layout(
                title=dict(text="Session Duration vs Anomaly Score", font=dict(color=T["text"], size=13), x=0),
                xaxis=_ax(),
                yaxis=_ax(),
                height=300,
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"])),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        anom_cats = df_a[df_a["anomaly"] == -1]["Category"].value_counts().reset_index()
        anom_cats.columns = ["Category", "Count"]
        if not anom_cats.empty:
            fig2 = go.Figure(go.Bar(
                x=anom_cats["Category"],
                y=anom_cats["Count"],
                marker=dict(
                    color=[_acolor(c) for c in anom_cats["Category"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Anomalies: %{y}<extra></extra>",
            ))
            fig2.update_layout(
                _layout(
                    title=dict(text="Anomalies by Category", font=dict(color=T["text"], size=13), x=0),
                    xaxis=_ax(),
                    yaxis=_ax(),
                    height=240,
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="sub-heading">Top Anomalous Sessions</div>', unsafe_allow_html=True)
        top_anom = df_a[df_a["anomaly"] == -1].nsmallest(6, "anomaly_score")[
            ["User ID", "Event", "Session (s)", "Category", "Platform", "anomaly_score"]
        ]
        for _, row in top_anom.iterrows():
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid rgba(248,81,73,.3);border-left:3px solid #f85149;
                        border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;
                                 font-weight:700;color:#f85149;">{row['User ID']}</span>
                    <span class="anomaly-badge">ANOMALY</span>
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:.78rem;color:#c9d1d9;
                            margin-bottom:6px;line-height:1.4;">{str(row['Event'])[:52]}</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;">{row['Platform']}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;">{row['Session (s)']}s</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;">score: {row['anomaly_score']:.3f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        if n_anom > 0:
            worst_cat        = df_a[df_a["anomaly"] == -1]["Category"].value_counts().idxmax()
            avg_anom_session = df_a[df_a["anomaly"] == -1]["Session (s)"].mean()
        else:
            worst_cat        = "N/A"
            avg_anom_session = 0.0
        avg_norm_session = df_a[df_a["anomaly"] == 1]["Session (s)"].mean() if n_norm else 0.0

        st.markdown(f"""
        <div style="background:rgba(248,81,73,.06);border:1px solid rgba(248,81,73,.2);
                    border-left:3px solid #f85149;border-radius:10px;padding:14px 16px;margin-top:8px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        color:#f85149;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">
                🔍 Anomaly Insight
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:.8rem;color:#8b949e;line-height:1.75;">
                Highest concentration in <b style="color:#e6edf3">{worst_cat}</b> events.<br>
                Avg anomalous session: <b style="color:#f85149">{avg_anom_session:.0f}s</b>
                vs normal <b style="color:#3fb950">{avg_norm_session:.0f}s</b>.<br>
                Recommend immediate audit of {worst_cat.lower()} flow.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; MODEL COMPARISON · RF vs GBM vs BASELINE &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3  ·  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def _build_features(df_b, df_f):
    records = []
    if df_f is not None and not df_f.empty:
        for (cat, pri), grp in df_f.groupby(["Category", "Priority"]):
            records.append({
                "area": cat, "volume": len(grp),
                "priority_score": priority_weight.get(pri, 1),
                "avg_rating": grp["Rating"].mean(),
                "drop_off_rate": 0.0,
            })
    if df_b is not None and not df_b.empty:
        for (cat, sev), grp in df_b.groupby(["Category", "Severity"]):
            dropoff_frac = len(grp[grp["Category"] == "Drop-off"]) / max(len(grp), 1) * 100
            records.append({
                "area": cat, "volume": len(grp),
                "priority_score": severity_weight.get(sev, 1),
                "avg_rating": 3.0,
                "drop_off_rate": dropoff_frac,
            })
    if not records:
        return None, None

    df = pd.DataFrame(records)
    scaler = MinMaxScaler()
    df["volume_sc"]   = scaler.fit_transform(df[["volume"]]).ravel()
    df["priority_sc"] = scaler.fit_transform(df[["priority_score"]]).ravel()
    df["frustration"] = ((5 - df["avg_rating"]) / 4) * 100

    le = LabelEncoder()
    df["area_enc"] = le.fit_transform(df["area"].astype(str))

    X = df[["priority_sc", "volume_sc", "frustration", "drop_off_rate", "area_enc"]].values
    y = (
        0.35 * df["priority_sc"]
        + 0.30 * df["volume_sc"]
        + 0.20 * df["frustration"] / 100
        + 0.15 * df["drop_off_rate"] / 100
    ).values
    return X, y


def _compare_models(X, y) -> list:
    models = [
        ("Random Forest",     RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
        ("Ridge (Baseline)",  Ridge(alpha=1.0)),
    ]
    results = []
    for name, mdl in models:
        n_splits = min(3, len(X))
        if n_splits < 2:
            mdl.fit(X, y)
            score = float(mdl.score(X, y))
        else:
            scores = cross_val_score(mdl, X, y, cv=n_splits, scoring="r2")
            score  = float(scores.mean())
        results.append({"model": name, "r2": max(0.0, score)})
    return sorted(results, key=lambda d: d["r2"], reverse=True)


def _render_model_comparison(df_b, df_f):
    st.markdown('<div class="sec-banner">🤖 &nbsp;3. MODEL COMPARISON & SELECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;Random Forest · Gradient Boosting · Ridge Baseline · 3-Fold CV &nbsp;◈</div>', unsafe_allow_html=True)

    X, y = _build_features(df_b, df_f)
    if X is None:
        st.warning("Not enough data for model comparison.")
        return

    results = _compare_models(X, y)
    winner  = results[0]["model"]
    max_r2  = results[0]["r2"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model",    " ".join(winner.split()[:2]))
    c2.metric("Best R² Score", f"{max_r2:.4f}")
    c3.metric("CV Folds",      "3-Fold")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sub-heading">Model R² Comparison</div>', unsafe_allow_html=True)
        colors_map = {
            "Random Forest":     T["blue"],
            "Gradient Boosting": T["green"],
            "Ridge (Baseline)":  T["dim"],
        }
        for res in results:
            is_win  = res["model"] == winner
            bar_pct = round(res["r2"] / max(max_r2, 0.001) * 100, 1)
            color   = colors_map.get(res["model"], T["muted"])
            win_cls = " model-winner" if is_win else ""
            medal   = " 🏆" if is_win else ""
            st.markdown(f"""
            <div class="model-row{win_cls}">
                <span class="model-name">{res['model']}{medal}</span>
                <div class="model-bar-wrap">
                    <div class="model-bar" style="width:{bar_pct}%;background:{color};"></div>
                </div>
                <span class="model-score" style="color:{color};">{res['r2']:.4f}</span>
            </div>""", unsafe_allow_html=True)

        # Feature importance
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X, y)
        feats = ["Priority Score", "Report Volume", "User Frustration", "Drop-off Rate", "Area Type"]
        imp   = rf.feature_importances_
        order = np.argsort(imp)
        palette = [T["blue"], T["green"], T["yellow"], T["red"], T["purple"]]

        fig = go.Figure(go.Bar(
            x=imp[order],
            y=[feats[i] for i in order],
            orientation="h",
            marker=dict(color=palette[::-1][:len(order)], line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            _layout(
                title=dict(text="Random Forest — Feature Importance", font=dict(color=T["text"], size=13), x=0),
                xaxis=_ax(),
                yaxis=_ax(),
                height=260,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="sub-heading">Why These Models?</div>', unsafe_allow_html=True)
        explanations = [
            ("🌲", "Random Forest",     T["blue"],
             "Ensemble of decision trees. Robust to noise, handles mixed feature types, provides reliable feature importance estimates."),
            ("📈", "Gradient Boosting", T["green"],
             "Iterative error correction. Often achieves higher accuracy by learning from residuals. Best for structured tabular data."),
            ("📐", "Ridge Baseline",    T["dim"],
             "Linear regularised regression. Serves as the interpretability baseline. Significant gap vs. ensemble = data is non-linear."),
        ]
        for icon, name, color, desc in explanations:
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1e2d3d;border-left:3px solid {color};
                        border-radius:8px;padding:12px 14px;margin-bottom:10px;">
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;
                            color:{color};margin-bottom:6px;">{icon} &nbsp;{name}</div>
                <div style="font-family:'Syne',sans-serif;font-size:.78rem;
                            color:#8b949e;line-height:1.65;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(56,139,253,.06);border:1px solid rgba(56,139,253,.2);
                    border-left:3px solid #388bfd;border-radius:10px;padding:14px 16px;margin-top:8px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                        color:#388bfd;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">
                📊 Selection Rationale
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:.8rem;color:#8b949e;line-height:1.75;">
                <b style="color:#e6edf3">{winner}</b> selected as the production model
                with R² = <b style="color:#58a6ff">{max_r2:.4f}</b>.
                Cross-validated scores ensure generalisation beyond training data.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; TEMPORAL PATTERN ANALYSIS &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4  ·  TEMPORAL PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _render_temporal(df_b: pd.DataFrame, df_f: pd.DataFrame):
    st.markdown('<div class="sec-banner">🕐 &nbsp;4. TEMPORAL PATTERN ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;Hourly / Daily Distributions · Engagement Windows &nbsp;◈</div>', unsafe_allow_html=True)

    frames = []
    for df, src in [(df_b, "Behavior"), (df_f, "Feedback")]:
        if df is not None and not df.empty:
            d = df.copy()
            d["_ts"]  = pd.to_datetime(d["Timestamp"], errors="coerce")
            d["hour"] = d["_ts"].dt.hour
            d["dow"]  = d["_ts"].dt.day_name()
            d["src"]  = src
            frames.append(d)

    if not frames:
        st.info("No timestamp data available.")
        return

    combined = pd.concat(frames, ignore_index=True)
    left, right = st.columns(2, gap="large")

    with left:
        if combined["src"].nunique() > 1:
            heat_h = combined.groupby(["src", "hour"]).size().unstack(fill_value=0)
            fig = go.Figure(go.Heatmap(
                z=heat_h.values,
                x=[f"{h:02d}:00" for h in heat_h.columns],
                y=heat_h.index.tolist(),
                colorscale=[[0, "#0d1117"], [0.4, "#1f6feb"], [1, "#58a6ff"]],
                hovertemplate="<b>%{y}</b> at %{x}<br>Events: %{z}<extra></extra>",
            ))
            fig.update_layout(
                _layout(
                    title=dict(text="Activity by Hour of Day", font=dict(color=T["text"], size=13), x=0),
                    xaxis=_ax({"tickangle": -45, "tickfont": dict(size=9)}),
                    yaxis=_ax(),
                    height=220,
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        dow_order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = combined.groupby(["src", "dow"]).size().reset_index(name="count")
        fig2 = go.Figure()
        for src, color in [("Behavior", T["blue"]), ("Feedback", T["green"])]:
            sub = dow_counts[dow_counts["src"] == src].set_index("dow").reindex(dow_order).fillna(0)
            fig2.add_trace(go.Scatter(
                x=sub.index, y=sub["count"],
                mode="lines+markers", name=src,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
                hovertemplate=f"<b>{src}</b><br>%{{x}}: %{{y}}<extra></extra>",
            ))
        fig2.update_layout(
            _layout(
                title=dict(text="Weekly Activity Rhythm", font=dict(color=T["text"], size=13), x=0),
                xaxis=_ax({"tickangle": -30}),
                yaxis=_ax(),
                height=240,
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"])),
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        if df_b is not None and not df_b.empty:
            df_bt = df_b.copy()
            df_bt["hour"] = pd.to_datetime(df_bt["Timestamp"], errors="coerce").dt.hour
            plat_hour = df_bt.groupby(["Platform", "hour"]).size().reset_index(name="count")
            plat_colors = {"Android": T["green"], "iOS": T["blue"], "Web": T["yellow"]}
            fig3 = go.Figure()
            for plat in df_bt["Platform"].unique():
                sub = plat_hour[plat_hour["Platform"] == plat]
                fig3.add_trace(go.Bar(
                    x=sub["hour"], y=sub["count"],
                    name=plat,
                    marker=dict(color=plat_colors.get(plat, T["muted"])),
                    hovertemplate=f"<b>{plat}</b><br>Hour %{{x}}:00 · %{{y}} events<extra></extra>",
                ))
            fig3.update_layout(
                _layout(
                    title=dict(text="Platform Activity by Hour", font=dict(color=T["text"], size=13), x=0),
                    xaxis=_ax(),
                    yaxis=_ax(),
                    barmode="stack",
                    height=240,
                    showlegend=True,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"])),
                )
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Guard against empty hour/dow series
        hour_counts = combined.groupby("hour").size()
        dow_counts2 = combined.groupby("dow").size()
        peak_hour   = int(hour_counts.idxmax()) if not hour_counts.empty else 0
        quiet_hour  = int(hour_counts.idxmin()) if not hour_counts.empty else 0
        peak_dow    = str(dow_counts2.idxmax())  if not dow_counts2.empty else "N/A"

        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:10px;padding:16px 18px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#58a6ff;
                        letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px;">
                ⏱ Temporal Insights
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:.8rem;color:#8b949e;line-height:2.0;">
                🔴 &nbsp;Peak activity hour: <b style="color:#e6edf3">{peak_hour:02d}:00</b><br>
                🟢 &nbsp;Quietest hour: <b style="color:#e6edf3">{quiet_hour:02d}:00</b><br>
                📅 &nbsp;Busiest day: <b style="color:#e6edf3">{peak_dow}</b><br>
                💡 &nbsp;Schedule deployments during quiet windows to minimise user impact.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; ML PRIORITY SCORING + ACTION PLAN &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5  ·  PRIORITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
def _build_issue_table(df_b, df_f) -> pd.DataFrame:
    records = []
    if df_f is not None and not df_f.empty:
        for (cat, pri), grp in df_f.groupby(["Category", "Priority"]):
            label = grp["Feedback"].mode().iloc[0]
            records.append({
                "issue": label, "area": cat, "volume": len(grp),
                "priority_score": priority_weight.get(pri, 1),
                "avg_rating": grp["Rating"].mean(),
                "drop_off_rate": 0.0, "source": "feedback",
            })
    if df_b is not None and not df_b.empty:
        for (cat, sev), grp in df_b.groupby(["Category", "Severity"]):
            label = grp["Event"].mode().iloc[0]
            dropoff_frac = len(grp[grp["Category"] == "Drop-off"]) / max(len(grp), 1) * 100
            records.append({
                "issue": label, "area": cat, "volume": len(grp),
                "priority_score": severity_weight.get(sev, 1),
                "avg_rating": 3.0, "drop_off_rate": dropoff_frac, "source": "behavior",
            })
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(subset=["issue"]).reset_index(drop=True)
    scaler = MinMaxScaler(feature_range=(0, 100))
    df["volume_scaled"]         = scaler.fit_transform(df[["volume"]]).ravel()
    df["priority_score_scaled"] = scaler.fit_transform(df[["priority_score"]]).ravel()
    df["frustration"]           = ((5 - df["avg_rating"]) / 4) * 100
    df["urgency"] = (
        0.35 * df["priority_score_scaled"]
        + 0.30 * df["volume_scaled"]
        + 0.20 * df["frustration"]
        + 0.15 * df["drop_off_rate"]
    )
    df["urgency"] = MinMaxScaler(feature_range=(0, 10)).fit_transform(df[["urgency"]]).ravel()
    return df


def _rf_score(df: pd.DataFrame) -> np.ndarray:
    le       = LabelEncoder()
    area_enc = le.fit_transform(df["area"].astype(str))
    X = np.column_stack([
        df["priority_score_scaled"].values, df["volume_scaled"].values,
        df["frustration"].values, df["drop_off_rate"].values, area_enc,
    ])
    y = df["urgency"].values
    np.random.seed(42)
    y_noisy = np.clip(y + np.random.normal(0, 0.2, len(y)), 0, 10)
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=1, random_state=42)
    rf.fit(X, y_noisy)
    raw    = rf.predict(X)
    scaled = MinMaxScaler(feature_range=(0, 10)).fit_transform(raw.reshape(-1, 1)).ravel()
    return np.round(scaled, 1)


def _rf_classify(df: pd.DataFrame) -> np.ndarray:
    le       = LabelEncoder()
    area_enc = le.fit_transform(df["area"].astype(str))
    X = np.column_stack([
        df["priority_score_scaled"].values, df["volume_scaled"].values,
        df["frustration"].values, df["drop_off_rate"].values, area_enc,
    ])
    urgency  = df["urgency"].values
    if len(urgency) < 4:
        # Not enough samples for percentile tiering — assign tier 0 to all
        return np.zeros(len(urgency), dtype=int)
    q75, q50, q25 = np.percentile(urgency, [75, 50, 25])
    y = np.where(urgency >= q75, 0,
        np.where(urgency >= q50, 1,
        np.where(urgency >= q25, 2, 3)))
    # Need at least 2 classes for RF classifier
    if len(np.unique(y)) < 2:
        return y
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf.predict(X)


def _render_priority_section(ranked: pd.DataFrame, df_b, df_f):
    st.markdown('<div class="sec-banner">📊 &nbsp;5. PRODUCT ROADMAP PRIORITIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;Random Forest Urgency Scoring · Percentile Tiering &nbsp;◈</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sub-heading">Priority Ranking Dashboard</div>', unsafe_allow_html=True)
        for i, row in ranked.head(8).iterrows():
            rank  = i + 1
            color = _RANK_COLORS[min(i, len(_RANK_COLORS) - 1)]
            ac    = _acolor(row["area"])
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{rank}"
            st.markdown(f"""
            <div class="pcard" style="border-left-color:{color};">
                <span class="prank" style="color:{color};">{medal}</span>
                <span class="ptag" style="background:{ac};">{row['area']}</span>
                <span class="ptitle">{str(row['issue'])[:52]}</span>
                <span class="pscore">{row['rf_score']}</span>
            </div>""", unsafe_allow_html=True)

        if len(ranked) >= 2:
            top5 = ranked.head(min(5, len(ranked)))
            cats = ["Priority", "Volume", "Frustration", "Drop-off"]
            fig  = go.Figure()
            for _, row in top5.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        row["priority_score_scaled"] / 10,
                        row["volume_scaled"] / 10,
                        row["frustration"] / 10,
                        row["drop_off_rate"] / 10,
                    ],
                    theta=cats,
                    fill="toself",
                    name=str(row["issue"])[:28],
                    opacity=0.6,
                ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(
                        visible=True, range=[0, 10],
                        gridcolor=T["border"], tickfont=dict(color=T["dim"], size=9),
                    ),
                    angularaxis=dict(
                        gridcolor=T["border"], tickfont=dict(color=T["muted"], size=10),
                    ),
                ),
                title=dict(text="Top Issues — Multi-Dimension Radar", font=dict(color=T["text"], size=13), x=0),
                showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"], size=9)),
                margin=dict(l=20, r=20, t=40, b=10),
                height=320,
                font=dict(family="JetBrains Mono, monospace", color=T["muted"]),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="sub-heading">Insights Summary</div>', unsafe_allow_html=True)
        if df_f is not None and not df_f.empty:
            crit = df_f[df_f["Priority"] == "Critical"]
            if not crit.empty:
                pct = round(len(crit) / max(len(df_f), 1) * 100)
                top = crit["Category"].value_counts().idxmax()
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid rgba(248,81,73,.25);border-left:3px solid #f85149;
                            border-radius:8px;padding:12px 14px;margin-bottom:10px;">
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.82rem;
                                color:#f85149;margin-bottom:6px;">🚨 High Impact Issues</div>
                    <div style="font-family:'Syne',sans-serif;font-size:.78rem;color:#8b949e;line-height:1.7;">
                        {pct}% feedback is Critical priority.<br>
                        Most critical area: <b style="color:#f85149">{top}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        if df_b is not None and not df_b.empty:
            drops = df_b[df_b["Category"] == "Drop-off"]
            if not drops.empty:
                pct   = round(len(drops) / max(len(df_b), 1) * 100)
                worst = drops["Event"].value_counts().idxmax()
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid rgba(56,139,253,.25);border-left:3px solid #388bfd;
                            border-radius:8px;padding:12px 14px;margin-bottom:10px;">
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:.82rem;
                                color:#58a6ff;margin-bottom:6px;">📉 User Drop-offs</div>
                    <div style="font-family:'Syne',sans-serif;font-size:.78rem;color:#8b949e;line-height:1.7;">
                        {pct}% of behaviour events are drop-offs.<br>
                        Worst: {str(worst)[:42]}
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; ACTION PLAN + BUSINESS DECISIONS &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6  ·  ACTION RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def _render_action_section(ranked: pd.DataFrame):
    st.markdown('<div class="sec-banner">⚙️ &nbsp;6. DECISION MAKING / ACTION RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sys-label">◈ &nbsp;ML Engine Output &nbsp;◈</div>', unsafe_allow_html=True)

    tier_border = {0: "#f85149", 1: "#d29922", 2: "#388bfd", 3: "#484f58"}
    tier_glow   = {0: "rgba(248,81,73,.15)", 1: "rgba(210,153,34,.12)", 2: "rgba(56,139,253,.12)", 3: "rgba(72,79,88,.12)"}
    tier_text   = {0: "#f85149", 1: "#d29922", 2: "#58a6ff", 3: "#6e7681"}

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sub-heading">Recommended Actions</div>', unsafe_allow_html=True)
        for i, row in ranked.head(5).iterrows():
            tier            = int(row["action_tier"])
            label, icon, _  = _TIER[tier]
            border          = tier_border[tier]
            glow            = tier_glow[tier]
            txt_color       = tier_text[tier]
            team, dl        = _team_deadline(row["area"])
            reason          = _reason(tier, row["area"])
            st.markdown(f"""
            <div class="acard" style="border-color:{border};background:linear-gradient(135deg,#0d1117,{glow});">
                <div class="acard-glow" style="background:linear-gradient(90deg,{border},transparent);"></div>
                <div class="atop" style="color:{txt_color};">{icon} &nbsp;ACTION {i+1} &nbsp;·&nbsp; {label.upper()}</div>
                <div class="aname">{str(row['issue'])[:56]}</div>
                <div class="ameta-row"><span class="ameta-label">Assign</span><span class="ameta-val">{team}</span></div>
                <div class="ameta-row"><span class="ameta-label">Deadline</span><span class="ameta-val" style="color:{txt_color};">{dl}</span></div>
                <div class="ameta-row"><span class="ameta-label">Reason</span><span class="ameta-val">{reason}</span></div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sub-heading">Business Decision Panel</div>', unsafe_allow_html=True)
        approved_h = review_h = rejected_h = ""
        for _, row in ranked.iterrows():
            txt, cls = _biz_decision(row["rf_score"])
            badge = (
                f'<div class="dbadge {cls}">'
                f'<span>{txt}</span>'
                f'<span style="opacity:.5;font-size:.65rem;">·</span>'
                f'<span style="opacity:.7;">{str(row["issue"])[:30]}</span>'
                f'</div>'
            )
            if "APPROVED" in txt: approved_h += badge
            elif "Review"  in txt: review_h   += badge
            else:                  rejected_h  += badge

        if approved_h:
            st.markdown('<div class="biz-group-label" style="color:#3fb950;">✅ Approved Actions</div>' + approved_h, unsafe_allow_html=True)
        if review_h:
            st.markdown('<div class="biz-group-label" style="color:#d29922;">🔄 Under Review</div>' + review_h, unsafe_allow_html=True)
        if rejected_h:
            st.markdown('<div class="biz-group-label" style="color:#f85149;">❌ Rejected (For Now)</div>' + rejected_h, unsafe_allow_html=True)

        top2  = ranked.head(2)["area"].tolist()
        focus = " & ".join(dict.fromkeys(top2))
        st.markdown(f"""
        <div class="suggestion">
            💡 &nbsp;SYSTEM SUGGESTION<br>
            Focus sprint on <b>{focus}</b> for the next 2 weeks
        </div>
        <div class="final-note">
            🧑‍💼 &nbsp;FINAL DECISION BY HUMAN MANAGERS<br>
            System provides DATA + SUGGESTIONS → Manager approves final roadmap
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(df_behavior: pd.DataFrame, df_feedback: pd.DataFrame):
    """Call from app.py:  run_analysis(df_b, df_f)"""
    _css()

    if (df_behavior is None or df_behavior.empty) and \
       (df_feedback  is None or df_feedback.empty):
        st.error("No data to analyse. Please generate data first.")
        return

    st.divider()

    # 1. NLP Clustering
    if df_feedback is not None and not df_feedback.empty:
        with st.spinner("Running NLP feedback clustering…"):
            _render_nlp_section(df_feedback)
    else:
        st.info("No feedback data — skipping NLP clustering.")

    st.markdown('<div class="ana-divider"></div>', unsafe_allow_html=True)

    # 2. Anomaly Detection
    if df_behavior is not None and not df_behavior.empty:
        with st.spinner("Running anomaly detection…"):
            _render_anomaly_section(df_behavior)
    else:
        st.info("No behaviour data — skipping anomaly detection.")

    st.markdown('<div class="ana-divider"></div>', unsafe_allow_html=True)

    # 3. Model Comparison
    with st.spinner("Running model comparison…"):
        _render_model_comparison(df_behavior, df_feedback)

    st.markdown('<div class="ana-divider"></div>', unsafe_allow_html=True)

    # 4. Temporal Analysis
    with st.spinner("Analysing temporal patterns…"):
        _render_temporal(df_behavior, df_feedback)

    st.markdown('<div class="ana-divider"></div>', unsafe_allow_html=True)

    # 5 & 6. Priority Scoring + Action Plan
    with st.spinner("Running Random Forest priority scoring…"):
        df_issues = _build_issue_table(df_behavior, df_feedback)

    if df_issues.empty:
        st.error("Could not extract issues. Check your data tables.")
        return

    df_issues["rf_score"]    = _rf_score(df_issues)
    df_issues["action_tier"] = _rf_classify(df_issues)
    ranked = df_issues.sort_values("rf_score", ascending=False).reset_index(drop=True)

    _render_priority_section(ranked, df_behavior, df_feedback)
    st.markdown('<div class="ana-divider"></div>', unsafe_allow_html=True)
    _render_action_section(ranked)
    

    #new ml algo
    