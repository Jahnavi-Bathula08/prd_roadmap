"""
ml_analysis.py
==============
Plug-and-play ML analysis for the Rule-Based Data Generator app.

Call  run_analysis(df_behavior, df_feedback)  from your app.py
when the Analyse button is clicked.

How to integrate into app.py
─────────────────────────────
1.  Place this file in the same folder as app.py
2.  Add at the top of app.py:
        from ml_analysis import run_analysis

3.  Add this block BELOW the existing "Clear Data" section at the bottom
    of your app.py  (inside the  `if data_exists:`  block):

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyse Data", use_container_width=True, type="primary"):
                run_analysis(df_b, df_f)

Dependencies (already standard with streamlit):
    pip install streamlit pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def _css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ── Section banner ─────────────────────────────────── */
    .sec-banner {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-left: 3px solid #58a6ff;
        color: #e6edf3;
        padding: 14px 20px;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 28px 0 6px;
        text-transform: uppercase;
    }

    .sys-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #388bfd;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 18px;
        padding-left: 2px;
    }

    .arrow-divider {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #388bfd;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 24px 0;
        padding: 10px;
        border: 1px dashed #1e2d3d;
        border-radius: 6px;
        background: rgba(56,139,253,0.04);
    }

    /* ── Priority ranking cards ─────────────────────────── */
    .pcard {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-left: 4px solid;
        transition: border-color 0.15s ease, background 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .pcard::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 60px; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.015));
        pointer-events: none;
    }
    .prank {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.82rem;
        min-width: 38px;
        letter-spacing: 0.04em;
    }
    .ptag {
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        color: #fff;
        white-space: nowrap;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .ptitle {
        flex: 1;
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #c9d1d9;
        line-height: 1.3;
    }
    .pscore {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.88rem;
        color: #58a6ff;
        background: rgba(56,139,253,0.1);
        border: 1px solid rgba(56,139,253,0.25);
        border-radius: 6px;
        padding: 3px 10px;
        white-space: nowrap;
    }

    /* ── Insight cards ──────────────────────────────────── */
    .ins {
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-family: 'Syne', sans-serif;
        font-size: 0.82rem;
        line-height: 1.75;
        border: 1px solid;
        background: #0d1117;
    }
    .ins-head {
        font-weight: 800;
        margin-bottom: 8px;
        font-size: 0.84rem;
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: 'Syne', sans-serif;
        letter-spacing: 0.02em;
    }
    .ins-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        color: #8b949e;
        padding: 2px 0;
    }
    .ins-dot {
        width: 4px; height: 4px;
        border-radius: 50%;
        margin-top: 7px;
        flex-shrink: 0;
    }

    /* ── Action cards ───────────────────────────────────── */
    .acard {
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid;
        background: #0d1117;
        position: relative;
        overflow: hidden;
    }
    .acard-glow {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .atop {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .aname {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 12px;
        line-height: 1.35;
    }
    .ameta-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #484f58;
        margin-bottom: 4px;
    }
    .ameta-label {
        color: #30363d;
        min-width: 52px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.65rem;
    }
    .ameta-val { color: #8b949e; }

    /* ── Business decision badges ───────────────────────── */
    .biz-group-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 14px 0 8px;
    }
    .dbadge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 8px;
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
        border: 1px solid;
    }
    .dapproved {
        background: rgba(63,185,80,0.07);
        color: #3fb950;
        border-color: rgba(63,185,80,0.25);
    }
    .dreview {
        background: rgba(210,153,34,0.07);
        color: #d29922;
        border-color: rgba(210,153,34,0.25);
    }
    .drejected {
        background: rgba(218,54,51,0.07);
        color: #f85149;
        border-color: rgba(218,54,51,0.2);
    }

    /* ── Suggestion & final note ────────────────────────── */
    .suggestion {
        background: rgba(210,153,34,0.07);
        border: 1px solid rgba(210,153,34,0.25);
        border-left: 3px solid #d29922;
        padding: 14px 16px;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: #d29922;
        margin-top: 16px;
        line-height: 1.7;
    }
    .suggestion b { color: #e3b341; }

    .final-note {
        background: rgba(56,139,253,0.06);
        border: 1px solid rgba(56,139,253,0.2);
        border-left: 3px solid #388bfd;
        padding: 12px 16px;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #58a6ff;
        margin-top: 10px;
        line-height: 1.8;
        letter-spacing: 0.02em;
    }

    /* ── Section sub-heading ────────────────────────────── */
    .sub-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #484f58;
        margin-bottom: 14px;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# Converts your two DataFrames into one "issue" row per topic cluster
# ══════════════════════════════════════════════════════════════════════════════
def _build_issue_table(df_b: pd.DataFrame, df_f: pd.DataFrame) -> pd.DataFrame:
    records = []

    priority_weight = {"Critical": 10, "High": 7, "Medium": 4, "Low": 1}
    severity_weight = {"High": 10, "Medium": 5, "Low": 1}

    # ── from User Feedback ────────────────────────────────────────────────────
    if df_f is not None and not df_f.empty:
        for (cat, pri), grp in df_f.groupby(["Category", "Priority"]):
            label = grp["Feedback"].mode().iloc[0]
            records.append({
                "issue":          label,
                "area":           cat,
                "volume":         len(grp),
                "priority_score": priority_weight.get(pri, 1),
                "avg_rating":     grp["Rating"].mean(),
                "drop_off_rate":  0.0,
                "source":         "feedback",
            })

    # ── from User Behavior Events ─────────────────────────────────────────────
    if df_b is not None and not df_b.empty:
        for (cat, sev), grp in df_b.groupby(["Category", "Severity"]):
            label = grp["Event"].mode().iloc[0]
            dropoff_frac = (
                len(grp[grp["Category"] == "Drop-off"]) / max(len(grp), 1) * 100
            )
            records.append({
                "issue":          label,
                "area":           cat,
                "volume":         len(grp),
                "priority_score": severity_weight.get(sev, 1),
                "avg_rating":     3.0,
                "drop_off_rate":  dropoff_frac,
                "source":         "behavior",
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(subset=["issue"]).reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0, 100))
    df["volume_scaled"]         = scaler.fit_transform(df[["volume"]]).ravel()
    df["priority_score_scaled"] = scaler.fit_transform(df[["priority_score"]]).ravel()

    # frustration: lower avg_rating → higher frustration
    df["frustration"] = ((5 - df["avg_rating"]) / 4) * 100

    df["urgency"] = (
        0.35 * df["priority_score_scaled"]
        + 0.30 * df["volume_scaled"]
        + 0.20 * df["frustration"]
        + 0.15 * df["drop_off_rate"]
    )
    df["urgency"] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
        df[["urgency"]]
    ).ravel()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST — PRIORITY SCORER  (Regressor)
# ══════════════════════════════════════════════════════════════════════════════
def _rf_score(df: pd.DataFrame) -> np.ndarray:
    le      = LabelEncoder()
    area_enc = le.fit_transform(df["area"].astype(str))

    X = np.column_stack([
        df["priority_score_scaled"].values,
        df["volume_scaled"].values,
        df["frustration"].values,
        df["drop_off_rate"].values,
        area_enc,
    ])
    y = df["urgency"].values

    np.random.seed(42)
    y_noisy = np.clip(y + np.random.normal(0, 0.2, len(y)), 0, 10)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=1, random_state=42
    )
    rf.fit(X, y_noisy)
    raw    = rf.predict(X)
    scaled = MinMaxScaler(feature_range=(0, 10)).fit_transform(
        raw.reshape(-1, 1)
    ).ravel()
    return np.round(scaled, 1)


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST — ACTION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
_TIER = {
    0: ("Immediate",     "⚡", "#c0392b"),
    1: ("High Priority", "🔥", "#e67e22"),
    2: ("Planned",       "📅", "#2980b9"),
    3: ("Backlog",       "🗂️",  "#7f8c8d"),
}

def _rf_classify(df: pd.DataFrame) -> np.ndarray:
    le       = LabelEncoder()
    area_enc = le.fit_transform(df["area"].astype(str))

    X = np.column_stack([
        df["priority_score_scaled"].values,
        df["volume_scaled"].values,
        df["frustration"].values,
        df["drop_off_rate"].values,
        area_enc,
    ])
    urgency = df["urgency"].values
    q75, q50, q25 = np.percentile(urgency, [75, 50, 25])
    y = np.where(urgency >= q75, 0,
        np.where(urgency >= q50, 1,
        np.where(urgency >= q25, 2, 3)))

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X, y)
    return clf.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_AREA_COLORS = {
    "bug":         "#e74c3c",
    "performance": "#8e44ad",
    "feature":     "#27ae60",
    "ux":          "#2980b9",
    "drop-off":    "#c0392b",
    "auth":        "#e67e22",
    "engagement":  "#16a085",
    "conversion":  "#1abc9c",
}

def _acolor(area: str) -> str:
    return _AREA_COLORS.get(str(area).lower().strip(), "#95a5a6")

_RANK_COLORS = ["#c0392b", "#e67e22", "#2980b9", "#27ae60", "#7f8c8d"]

def _team_deadline(area: str) -> tuple:
    a = area.lower()
    if "bug"         in a: return "Backend Team",  "48 Hours"
    if "performance" in a: return "Infra Team",    "3 Days"
    if "ux"          in a: return "UX/UI Team",    "1 Week"
    if "feature"     in a: return "Product Team",  "Next Sprint"
    if "drop"        in a: return "Growth Team",   "1 Week"
    if "auth"        in a: return "Security Team", "48 Hours"
    if "engagement"  in a: return "Marketing Team","2 Weeks"
    if "conversion"  in a: return "Sales Team",    "1 Week"
    return "Product Team", "2 Weeks"

def _reason(tier: int, area: str) -> str:
    a = area.lower()
    if tier == 0:           return "Critical Impact"
    if "bug"  in a:         return "High Severity Bug"
    if "drop" in a:         return "High Drop-off Rate"
    if "ux"   in a:         return "Poor User Experience"
    return "User Demand"

def _biz_decision(score: float) -> tuple:
    if score >= 7.5: return "✅ APPROVED",       "dapproved"
    if score >= 5.5: return "🔄 Under Review",    "dreview"
    return "❌ Rejected (For Now)",               "drejected"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  —  Priority Dashboard + Insights
# ══════════════════════════════════════════════════════════════════════════════
def _render_section1(ranked: pd.DataFrame, df_b, df_f):
    st.markdown(
        '<div class="sec-banner">📊 &nbsp;1. PRODUCT ROADMAP PRIORITIZATION</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sys-label">◈ &nbsp;ML Engine Output &nbsp;◈</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    # ── Left: ranking cards ───────────────────────────────────────────────────
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
                <span class="ptitle">{row['issue'][:52]}</span>
                <span class="pscore">{row['rf_score']}</span>
            </div>""", unsafe_allow_html=True)

    # ── Right: insights ───────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="sub-heading">Insights Summary</div>', unsafe_allow_html=True)

        # High Impact Issues
        hi = []
        if df_f is not None and not df_f.empty:
            crit = df_f[df_f["Priority"] == "Critical"]
            if not crit.empty:
                pct  = round(len(crit) / len(df_f) * 100)
                top  = crit["Category"].value_counts().idxmax()
                hi  += [f"{pct}% feedback is Critical priority",
                        f"Most critical area: <b style='color:#f85149'>{top}</b>"]
        if hi:
            items = "".join(f'<div class="ins-item"><div class="ins-dot" style="background:#f85149;"></div><div>{x}</div></div>' for x in hi)
            st.markdown(f"""
            <div class="ins" style="border-color:rgba(248,81,73,0.25);border-left:3px solid #f85149;">
                <div class="ins-head" style="color:#f85149;">🚨 &nbsp;High Impact Issues</div>
                {items}
            </div>""", unsafe_allow_html=True)

        # Top Requests
        req = []
        if df_f is not None and not df_f.empty:
            for cat, grp in df_f[df_f["Category"].isin(["Feature","UX"])].groupby("Category"):
                pct = round(len(grp) / len(df_f) * 100)
                req.append(f"{cat} &rarr; {pct}% of feedback")
        if req:
            items = "".join(f'<div class="ins-item"><div class="ins-dot" style="background:#d29922;"></div><div>{x}</div></div>' for x in req)
            st.markdown(f"""
            <div class="ins" style="border-color:rgba(210,153,34,0.25);border-left:3px solid #d29922;">
                <div class="ins-head" style="color:#d29922;">🏆 &nbsp;Top Requests</div>
                {items}
            </div>""", unsafe_allow_html=True)

        # Drop-offs
        drop = []
        if df_b is not None and not df_b.empty:
            drops = df_b[df_b["Category"] == "Drop-off"]
            if not drops.empty:
                pct  = round(len(drops) / len(df_b) * 100)
                worst = drops["Event"].value_counts().idxmax()
                drop += [f"{pct}% of behavior events are drop-offs",
                         f"Worst: {worst[:42]}"]
        if drop:
            items = "".join(f'<div class="ins-item"><div class="ins-dot" style="background:#388bfd;"></div><div>{x}</div></div>' for x in drop)
            st.markdown(f"""
            <div class="ins" style="border-color:rgba(56,139,253,0.25);border-left:3px solid #388bfd;">
                <div class="ins-head" style="color:#58a6ff;">📉 &nbsp;User Drop-offs</div>
                {items}
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ &nbsp; SYSTEM GENERATES ACTION PLAN &nbsp; ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  —  Action Recommendations + Business Panel
# ══════════════════════════════════════════════════════════════════════════════
def _render_section2(ranked: pd.DataFrame):
    st.markdown(
        '<div class="sec-banner">⚙️ &nbsp;2. DECISION MAKING / ACTION RECOMMENDATIONS</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sys-label">◈ &nbsp;ML Engine Output &nbsp;◈</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    tier_border = {0: "#f85149", 1: "#d29922", 2: "#388bfd", 3: "#484f58"}
    tier_glow   = {0: "rgba(248,81,73,0.15)", 1: "rgba(210,153,34,0.12)", 2: "rgba(56,139,253,0.12)", 3: "rgba(72,79,88,0.12)"}
    tier_text   = {0: "#f85149", 1: "#d29922", 2: "#58a6ff", 3: "#6e7681"}

    # ── Left: action cards ────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="sub-heading">Recommended Actions</div>', unsafe_allow_html=True)
        for i, row in ranked.head(5).iterrows():
            tier          = int(row["action_tier"])
            label, icon, _ = _TIER[tier]
            border        = tier_border[tier]
            glow          = tier_glow[tier]
            txt_color     = tier_text[tier]
            team, dl      = _team_deadline(row["area"])
            reason        = _reason(tier, row["area"])

            st.markdown(f"""
            <div class="acard" style="border-color:{border};background:linear-gradient(135deg,#0d1117,{glow});">
                <div class="acard-glow" style="background:linear-gradient(90deg,{border},transparent);"></div>
                <div class="atop" style="color:{txt_color};">{icon} &nbsp;ACTION {i+1} &nbsp;·&nbsp; {label.upper()}</div>
                <div class="aname">{row['issue'][:56]}</div>
                <div class="ameta-row">
                    <span class="ameta-label">Assign</span>
                    <span class="ameta-val">{team}</span>
                </div>
                <div class="ameta-row">
                    <span class="ameta-label">Deadline</span>
                    <span class="ameta-val" style="color:{txt_color};">{dl}</span>
                </div>
                <div class="ameta-row">
                    <span class="ameta-label">Reason</span>
                    <span class="ameta-val">{reason}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Right: business decision panel ───────────────────────────────────────
    with right:
        st.markdown('<div class="sub-heading">Business Decision Panel</div>', unsafe_allow_html=True)

        approved_h = review_h = rejected_h = ""
        for _, row in ranked.iterrows():
            txt, cls = _biz_decision(row["rf_score"])
            badge = f'<div class="dbadge {cls}"><span>{txt}</span><span style="opacity:.5;font-size:.65rem;">·</span><span style="opacity:.7;">{row["issue"][:30]}</span></div>'
            if "APPROVED" in txt: approved_h += badge
            elif "Review"  in txt: review_h   += badge
            else:                  rejected_h  += badge

        if approved_h:
            st.markdown(
                '<div class="biz-group-label" style="color:#3fb950;">✅ &nbsp;Approved Actions</div>' + approved_h,
                unsafe_allow_html=True,
            )
        if review_h:
            st.markdown(
                '<div class="biz-group-label" style="color:#d29922;">🔄 &nbsp;Under Review</div>' + review_h,
                unsafe_allow_html=True,
            )
        if rejected_h:
            st.markdown(
                '<div class="biz-group-label" style="color:#f85149;">❌ &nbsp;Rejected (For Now)</div>' + rejected_h,
                unsafe_allow_html=True,
            )

        # System suggestion
        top2  = ranked.head(2)["area"].tolist()
        focus = " &amp; ".join(dict.fromkeys(top2))

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
# FEATURE IMPORTANCE EXPANDER
# ══════════════════════════════════════════════════════════════════════════════
def _render_importance(df: pd.DataFrame):
    with st.expander("🤖  Random Forest — Feature Importance Breakdown"):
        le       = LabelEncoder()
        area_enc = le.fit_transform(df["area"].astype(str))
        X = np.column_stack([
            df["priority_score_scaled"].values,
            df["volume_scaled"].values,
            df["frustration"].values,
            df["drop_off_rate"].values,
            area_enc,
        ])
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, df["urgency"].values)

        imp_df = pd.DataFrame({
            "Signal":     ["Priority Score", "Report Volume",
                           "User Frustration", "Drop-off Rate", "Area Type"],
            "Importance %": np.round(rf.feature_importances_ * 100, 2),
        }).sort_values("Importance %", ascending=False)

        # Render as custom dark bars instead of plain dataframe
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:#484f58;letter-spacing:0.14em;text-transform:uppercase;
                    margin-bottom:14px;">Signal Contribution to ML Score</div>
        """, unsafe_allow_html=True)

        max_val = imp_df["Importance %"].max()
        bar_colors = ["#388bfd", "#3fb950", "#d29922", "#f85149", "#8b949e"]
        for idx, (_, r) in enumerate(imp_df.iterrows()):
            pct   = r["Importance %"]
            width = round(pct / max_val * 100, 1)
            color = bar_colors[idx % len(bar_colors)]
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:5px;">
                    <span style="font-family:'Syne',sans-serif;font-size:0.82rem;
                                 font-weight:600;color:#c9d1d9;">{r['Signal']}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                                 color:{color};font-weight:700;">{pct}%</span>
                </div>
                <div style="height:6px;background:#1e2d3d;border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{width}%;background:{color};
                                border-radius:3px;transition:width 0.3s ease;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                    color:#30363d;margin-top:12px;letter-spacing:0.04em;">
            Higher % → that signal drives the ML prioritisation score more strongly.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(df_behavior: pd.DataFrame, df_feedback: pd.DataFrame):
    """
    Main entry — call this from app.py:

        from ml_analysis import run_analysis
        run_analysis(df_b, df_f)
    """
    _css()

    if (df_behavior is None or df_behavior.empty) and \
       (df_feedback  is None or df_feedback.empty):
        st.error("No data to analyse. Please generate data first.")
        return

    with st.spinner("Running Random Forest analysis…"):
        df_issues = _build_issue_table(df_behavior, df_feedback)

    if df_issues.empty:
        st.error("Could not extract issues. Check your data tables.")
        return

    df_issues["rf_score"]    = _rf_score(df_issues)
    df_issues["action_tier"] = _rf_classify(df_issues)
    ranked = df_issues.sort_values("rf_score", ascending=False).reset_index(drop=True)

    st.divider()
    _render_section1(ranked, df_behavior, df_feedback)
    _render_section2(ranked)
    _render_importance(df_issues)