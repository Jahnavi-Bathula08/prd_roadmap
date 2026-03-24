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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .sec-banner {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #fff;
        padding: 12px 22px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: .6px;
        margin: 6px 0 4px;
    }
    .sys-label {
        text-align: center;
        color: #e74c3c;
        font-weight: 700;
        font-size: .82rem;
        margin-bottom: 12px;
        letter-spacing: .5px;
    }
    .arrow-divider {
        text-align: center;
        font-size: .9rem;
        font-weight: 700;
        color: #e74c3c;
        margin: 10px 0;
    }

    .pcard {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        border-radius: 10px;
        margin-bottom: 7px;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
        border-left: 5px solid;
    }
    .prank  { font-weight: 800; font-size: .95rem; min-width: 36px; }
    .ptag   { padding: 2px 9px; border-radius: 20px; font-size: .68rem;
              font-weight: 700; color: #fff; white-space: nowrap; }
    .ptitle { flex:1; font-size: .85rem; font-weight: 600; color: #1a1a2e; }
    .pscore { font-weight: 800; font-size: 1rem; color: #fff;
              background: #1a1a2e; border-radius: 6px; padding: 2px 10px; }

    .ins {
        border-radius: 9px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: .82rem;
        line-height: 1.65;
    }
    .ins-head { font-weight: 800; margin-bottom: 4px; font-size: .88rem; }

    .acard {
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #fff;
        box-shadow: 0 3px 10px rgba(0,0,0,.18);
    }
    .atop  { font-weight: 800; font-size: .88rem; margin-bottom: 3px; }
    .aname { font-size: 1rem; font-weight: 800; margin-bottom: 6px; }
    .ameta { font-size: .8rem; line-height: 1.8; }

    .dbadge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .dapproved { background:#d5f5e3; color:#1e8449; border:1.5px solid #27ae60; }
    .dreview   { background:#fef9e7; color:#9a7d0a; border:1.5px solid #f1c40f; }
    .drejected { background:#fadbd8; color:#922b21; border:1.5px solid #e74c3c; }

    .suggestion {
        background: linear-gradient(135deg,#fff8e1,#ffecb3);
        border-left: 4px solid #f9a825;
        padding: 10px 14px; border-radius: 8px;
        font-size: .82rem; font-weight: 700;
        color: #6d4c00; margin-top: 12px;
    }
    .final-note {
        background: #e8f4fd;
        border-left: 4px solid #2980b9;
        padding: 9px 13px; border-radius: 8px;
        font-size: .76rem; color: #1a5276; margin-top: 8px;
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
        '<div class="sec-banner">📊 1. PRODUCT ROADMAP PRIORITIZATION</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sys-label">⟨ System Output ⟩</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    # ── Left: ranking cards ───────────────────────────────────────────────────
    with left:
        st.markdown("**📋 Priority Ranking Dashboard**")
        for i, row in ranked.head(8).iterrows():
            rank  = i + 1
            color = _RANK_COLORS[min(i, len(_RANK_COLORS) - 1)]
            star  = "⭐" if rank <= 3 else ""
            ac    = _acolor(row["area"])
            st.markdown(f"""
            <div class="pcard" style="border-color:{color}">
                <span class="prank" style="color:{color}">{star}P{rank}</span>
                <span class="ptag"  style="background:{ac}">{row['area']}</span>
                <span class="ptitle">{row['issue'][:50]}</span>
                <span class="pscore">{row['rf_score']}</span>
            </div>""", unsafe_allow_html=True)

    # ── Right: insights ───────────────────────────────────────────────────────
    with right:
        st.markdown("**🔍 Insights Summary**")

        # High Impact Issues
        hi = []
        if df_f is not None and not df_f.empty:
            crit = df_f[df_f["Priority"] == "Critical"]
            if not crit.empty:
                pct  = round(len(crit) / len(df_f) * 100)
                top  = crit["Category"].value_counts().idxmax()
                hi  += [f"{pct}% feedback is Critical priority",
                        f"Most critical area: <b>{top}</b>"]
        if hi:
            st.markdown(f"""
            <div class="ins" style="background:#fde8e8;border-left:4px solid #e74c3c">
                <div class="ins-head" style="color:#c0392b">🚨 High Impact Issues</div>
                {"".join(f"<div>• {x}</div>" for x in hi)}
            </div>""", unsafe_allow_html=True)

        # Top Requests
        req = []
        if df_f is not None and not df_f.empty:
            for cat, grp in df_f[df_f["Category"].isin(["Feature","UX"])].groupby("Category"):
                pct = round(len(grp) / len(df_f) * 100)
                req.append(f"{cat} → {pct}% of feedback")
        if req:
            st.markdown(f"""
            <div class="ins" style="background:#fef9e7;border-left:4px solid #f1c40f">
                <div class="ins-head" style="color:#9a7d0a">🏆 Top Requests</div>
                {"".join(f"<div>• {x}</div>" for x in req)}
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
            st.markdown(f"""
            <div class="ins" style="background:#e8f4fd;border-left:4px solid #2980b9">
                <div class="ins-head" style="color:#1a5276">📉 User Drop-offs</div>
                {"".join(f"<div>• {x}</div>" for x in drop)}
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="arrow-divider">⬇ System Generates Priority List ⬇</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  —  Action Recommendations + Business Panel
# ══════════════════════════════════════════════════════════════════════════════
def _render_section2(ranked: pd.DataFrame):
    st.markdown(
        '<div class="sec-banner">⚙️ 2. DECISION MAKING / ACTION RECOMMENDATIONS</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sys-label">⟨ System Output ⟩</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    tier_bg = {0: "#c0392b", 1: "#e67e22", 2: "#2980b9", 3: "#7f8c8d"}

    # ── Left: action cards ────────────────────────────────────────────────────
    with left:
        st.markdown("**✅ Recommended Actions**")
        for i, row in ranked.head(5).iterrows():
            tier          = int(row["action_tier"])
            label, icon, _ = _TIER[tier]
            color         = tier_bg[tier]
            team, dl      = _team_deadline(row["area"])
            reason        = _reason(tier, row["area"])

            st.markdown(f"""
            <div class="acard" style="background:{color}">
                <div class="atop">{icon} ACTION {i+1} ({label})</div>
                <div class="aname">{row['issue'][:54]}</div>
                <div class="ameta">
                    📌 Assign to: {team}<br>
                    ⏰ Deadline: {dl}<br>
                    💡 Reason: {reason}
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Right: business decision panel ───────────────────────────────────────
    with right:
        st.markdown("**🗂️ Business Decision Panel**")

        approved_h = review_h = rejected_h = ""
        for _, row in ranked.iterrows():
            txt, cls = _biz_decision(row["rf_score"])
            badge = (
                f'<div class="dbadge {cls}">'
                f'{txt} — {row["issue"][:28]}</div><br>'
            )
            if "APPROVED" in txt: approved_h += badge
            elif "Review"  in txt: review_h   += badge
            else:                  rejected_h  += badge

        if approved_h:
            st.markdown(
                '<div style="font-weight:800;color:#1e8449;margin-bottom:4px">'
                'Approved Actions</div>' + approved_h,
                unsafe_allow_html=True,
            )
        if review_h:
            st.markdown(
                '<div style="font-weight:800;color:#9a7d0a;margin:8px 0 4px">'
                'Under Review</div>' + review_h,
                unsafe_allow_html=True,
            )
        if rejected_h:
            st.markdown(
                '<div style="font-weight:800;color:#922b21;margin:8px 0 4px">'
                'Rejected (For Now)</div>' + rejected_h,
                unsafe_allow_html=True,
            )

        # System suggestion
        top2  = ranked.head(2)["area"].tolist()
        focus = " & ".join(dict.fromkeys(top2))

        st.markdown(f"""
        <div class="suggestion">
            💡 SYSTEM SUGGESTION:<br>
            Focus on <b>{focus}</b> for the next 2 weeks
        </div>
        <div class="final-note">
            🧑‍💼 <b>Final Decision by Human Managers</b><br>
            System provides DATA + SUGGESTIONS → Manager approves final roadmap
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE EXPANDER
# ══════════════════════════════════════════════════════════════════════════════
def _render_importance(df: pd.DataFrame):
    with st.expander("🤖 Random Forest — Feature Importance Breakdown"):
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

        st.dataframe(imp_df, use_container_width=True, hide_index=True)
        st.caption(
            "Higher % → that signal drives the ML prioritisation score more strongly."
        )


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