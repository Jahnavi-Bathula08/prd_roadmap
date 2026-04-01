import streamlit as st
import pandas as pd
import random
import json
import os
from datetime import datetime, timedelta
from ml_analysis import run_analysis          # ← ML module

# ─── Persistence File Path ────────────────────────────────────────────────────
SAVE_FILE = "generated_data.json"

# ─── Rule-based Templates ─────────────────────────────────────────────────────

BEHAVIOR_TEMPLATES = [
    {"variations": ["User dropped at payment page", "User exited on payment screen", "User left during payment step"], "category": "Drop-off", "severity": "High"},
    {"variations": ["User clicked add to cart", "User added item to cart", "User tapped add to cart button"], "category": "Engagement", "severity": "Low"},
    {"variations": ["User failed login attempt", "User couldn't log in", "User login rejected"], "category": "Auth", "severity": "Medium"},
    {"variations": ["User abandoned signup flow", "User quit during registration", "User dropped off at signup"], "category": "Drop-off", "severity": "High"},
    {"variations": ["User viewed product details", "User opened product page", "User browsed product info"], "category": "Engagement", "severity": "Low"},
    {"variations": ["User applied promo code", "User entered discount code", "User redeemed coupon"], "category": "Conversion", "severity": "Low"},
    {"variations": ["User removed item from cart", "User deleted product from cart", "User cleared cart item"], "category": "Drop-off", "severity": "Medium"},
    {"variations": ["User completed checkout", "User placed order successfully", "User finished purchase"], "category": "Conversion", "severity": "Low"},
    {"variations": ["User searched for product", "User used search bar", "User typed in search field"], "category": "Engagement", "severity": "Low"},
    {"variations": ["User session timed out", "User was auto-logged out", "User idle session expired"], "category": "Auth", "severity": "Medium"},
    {"variations": ["User clicked on banner ad", "User tapped promotional banner", "User opened ad link"], "category": "Engagement", "severity": "Low"},
    {"variations": ["User skipped onboarding", "User dismissed tutorial", "User exited intro screen"], "category": "Drop-off", "severity": "Medium"},
]

FEEDBACK_TEMPLATES = [
    {"variations": ["Payment not working", "Unable to complete payment", "Payment page keeps failing"], "category": "Bug", "priority": "Critical"},
    {"variations": ["App is slow", "App takes too long to load", "Everything is lagging"], "category": "Performance", "priority": "High"},
    {"variations": ["Need dark mode", "Please add dark theme", "Dark mode is missing"], "category": "Feature", "priority": "Medium"},
    {"variations": ["Too many ads", "Ads are very annoying", "Please reduce ads"], "category": "UX", "priority": "Medium"},
    {"variations": ["Login keeps failing", "Can't sign in at all", "Login button not responding"], "category": "Bug", "priority": "Critical"},
    {"variations": ["Images not loading", "Product photos won't show", "Broken images everywhere"], "category": "Bug", "priority": "High"},
    {"variations": ["Want push notifications", "Add order update alerts", "Need notification support"], "category": "Feature", "priority": "Low"},
    {"variations": ["Checkout flow is confusing", "Too many steps to buy", "Hard to complete purchase"], "category": "UX", "priority": "High"},
    {"variations": ["Price filter not working", "Filter resets randomly", "Can't sort by price"], "category": "Bug", "priority": "Medium"},
    {"variations": ["Add wishlist feature", "Need a save-for-later option", "Want to bookmark products"], "category": "Feature", "priority": "Low"},
    {"variations": ["Search results are irrelevant", "Search is not accurate", "Wrong products showing in search"], "category": "Bug", "priority": "High"},
    {"variations": ["App crashes on startup", "App force-closes randomly", "Frequent unexpected crashes"], "category": "Bug", "priority": "Critical"},
]

PLATFORMS  = ["Android", "iOS", "Web"]
REGIONS    = ["North", "South", "East", "West", "Central"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# ─── Save / Load helpers ──────────────────────────────────────────────────────

def save_data(df_behavior, df_feedback, generated_at, n_rows):
    payload = {
        "generated_at": generated_at,
        "n_rows": n_rows,
        "behavior": df_behavior.to_dict(orient="records") if df_behavior is not None else None,
        "feedback": df_feedback.to_dict(orient="records") if df_feedback is not None else None,
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(payload, f)

def load_data():
    if not os.path.exists(SAVE_FILE):
        return None, None, None, None
    with open(SAVE_FILE, "r") as f:
        payload = json.load(f)
    df_b = pd.DataFrame(payload["behavior"]) if payload.get("behavior") else None
    df_f = pd.DataFrame(payload["feedback"]) if payload.get("feedback") else None
    return df_b, df_f, payload.get("generated_at"), payload.get("n_rows")

def clear_saved_data():
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)

# ─── Generators ───────────────────────────────────────────────────────────────

def random_timestamp(days_back=30):
    base = datetime.now() - timedelta(days=days_back)
    return base + timedelta(days=random.randint(0, days_back), hours=random.randint(0, 23), minutes=random.randint(0, 59))

def generate_behavior_data(n=50):
    rows = []
    for _ in range(n):
        t = random.choice(BEHAVIOR_TEMPLATES)
        rows.append({
            "User ID":     f"U{random.randint(1000, 9999)}",
            "Timestamp":   random_timestamp().strftime("%Y-%m-%d %H:%M"),
            "Event":       random.choice(t["variations"]),
            "Category":    t["category"],
            "Severity":    t["severity"],
            "Platform":    random.choice(PLATFORMS),
            "Region":      random.choice(REGIONS),
            "Session (s)": random.randint(5, 600),
        })
    return pd.DataFrame(rows)

def generate_feedback_data(n=50):
    rows = []
    for _ in range(n):
        t = random.choice(FEEDBACK_TEMPLATES)
        rows.append({
            "User ID":   f"U{random.randint(1000, 9999)}",
            "Timestamp": random_timestamp().strftime("%Y-%m-%d %H:%M"),
            "Feedback":  random.choice(t["variations"]),
            "Category":  t["category"],
            "Priority":  t["priority"],
            "Age Group": random.choice(AGE_GROUPS),
            "Platform":  random.choice(PLATFORMS),
            "Rating":    random.randint(1, 5),
        })
    return pd.DataFrame(rows)

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }

    .stApp { background: #080c14 !important; }

    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1e2d3d !important;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    .stSlider > div > div > div > div { background: #58a6ff !important; }
    .stCheckbox label { color: #8b949e !important; font-size: 0.88rem !important; }

    #MainMenu, footer, header { visibility: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #30363d !important;
        color: #8b949e !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
        padding: 10px 20px !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        background: rgba(88,166,255,0.06) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg,#1f6feb,#388bfd) !important;
        border: none !important;
        color: #fff !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(56,139,253,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg,#388bfd,#58a6ff) !important;
        box-shadow: 0 0 30px rgba(88,166,255,0.45) !important;
    }
    .stButton > button[kind="secondary"] {
        border-color: #da3633 !important;
        color: #da3633 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(218,54,51,0.08) !important;
    }
    .stDownloadButton > button {
        background: rgba(46,160,67,0.08) !important;
        border: 1px solid #2ea043 !important;
        color: #3fb950 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        background: #0d1117 !important;
        border: 1px solid #1e2d3d !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    hr { border: none !important; border-top: 1px solid #1e2d3d !important; margin: 28px 0 !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ── Donut card animations ── */
    @keyframes countUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes drawRing {
        from { stroke-dashoffset: 251; }
    }
    .metric-ring { animation: drawRing 1.2s cubic-bezier(.4,0,.2,1) forwards; }
    .metric-value { animation: countUp 0.6s ease forwards; }
    @keyframes barGrow {
        from { width: 0 !important; }
    }
    .bar-fill { animation: barGrow 1s cubic-bezier(.4,0,.2,1) forwards; }
    @keyframes chipPop {
        from { opacity:0; transform: scale(0.85); }
        to   { opacity:1; transform: scale(1); }
    }
    .chip-pop { animation: chipPop 0.4s ease forwards; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div style="padding:48px 0 36px 0;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1f6feb,#3fb950);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:22px;flex-shrink:0;">📊</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;
                            color:#e6edf3;letter-spacing:-0.02em;line-height:1;">DataForge Studio</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#58a6ff;
                            letter-spacing:0.15em;text-transform:uppercase;margin-top:5px;">
                    Rule-Based Synthetic Data Generator &nbsp;·&nbsp; ML Analysis Engine</div>
            </div>
        </div>
        <div style="width:100%;height:1px;background:linear-gradient(90deg,#1f6feb 0%,#3fb950 40%,transparent 100%);
                    opacity:0.5;margin-top:24px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar(generated_at, n_rows_used):
    st.markdown(f"""
    <div style="background:rgba(63,185,80,0.06);border:1px solid rgba(63,185,80,0.25);
                border-radius:10px;padding:12px 20px;display:flex;align-items:center;
                gap:12px;margin-bottom:24px;">
        <div style="width:8px;height:8px;background:#3fb950;border-radius:50%;
                    box-shadow:0 0 8px rgba(63,185,80,0.6);flex-shrink:0;"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                     color:#3fb950;letter-spacing:0.04em;">
            DATASET LOADED — {n_rows_used} rows/table · Generated {generated_at}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_section_heading(icon, title, subtitle=None):
    sub_html = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;color:#484f58;letter-spacing:0.08em;text-transform:uppercase;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:32px 0 20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;
                         color:#e6edf3;letter-spacing:-0.01em;">{title}</span>
        </div>
        {sub_html}
        <div style="width:40px;height:2px;background:linear-gradient(90deg,#58a6ff,transparent);
                    margin-top:10px;border-radius:1px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;border:1px dashed #1e2d3d;
                border-radius:16px;background:rgba(13,17,23,0.5);margin:40px 0;">
        <div style="font-size:3rem;margin-bottom:16px;opacity:0.4;">⚡</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                    color:#30363d;margin-bottom:8px;">No data generated yet</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                    color:#21262d;letter-spacing:0.05em;">
            Configure settings in the sidebar, then click Generate Data</div>
    </div>
    """, unsafe_allow_html=True)


# ─── ADVANCED DASHBOARD COMPONENTS ───────────────────────────────────────────

def _donut_ring_svg(value, total, color, size=80):
    """Returns an SVG donut ring showing value/total as a filled arc."""
    pct    = (value / total * 100) if total > 0 else 0
    radius = 34
    circ   = 2 * 3.14159 * radius   # ≈ 213.6
    filled = circ * pct / 100
    gap    = circ - filled
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 80 80" style="display:block;">
      <circle cx="40" cy="40" r="{radius}"
              fill="none" stroke="#1e2d3d" stroke-width="7"/>
      <circle cx="40" cy="40" r="{radius}"
              fill="none" stroke="{color}" stroke-width="7"
              stroke-linecap="round"
              stroke-dasharray="{filled:.1f} {gap:.1f}"
              stroke-dashoffset="53.4"
              class="metric-ring"
              style="transform-origin:center;transform:rotate(-90deg) translateX(-80px);
                     animation-duration:1.4s;"/>
    </svg>"""


def render_behavior_dashboard(df):
    total   = len(df)
    high    = len(df[df["Severity"] == "High"])
    drops   = len(df[df["Category"] == "Drop-off"])
    convs   = len(df[df["Category"] == "Conversion"])

    # ── Donut metric cards ─────────────────────────────────────────────────────
    cards = [
        {"label": "Total Events",  "value": total,  "sub": "all events",      "color": "#58a6ff", "pct": 100},
        {"label": "High Severity", "value": high,   "sub": "of total",        "color": "#f85149", "pct": round(high/total*100) if total else 0},
        {"label": "Drop-offs",     "value": drops,  "sub": "funnel exits",    "color": "#d29922", "pct": round(drops/total*100) if total else 0},
        {"label": "Conversions",   "value": convs,  "sub": "completed goals", "color": "#3fb950", "pct": round(convs/total*100) if total else 0},
    ]

    cols = st.columns(4)
    for col, c in zip(cols, cards):
        svg = _donut_ring_svg(c["value"], total, c["color"])
        col.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:14px;
                    padding:20px 18px;position:relative;overflow:hidden;
                    transition:border-color 0.2s ease;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,{c['color']},transparent);"></div>
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="flex-shrink:0;">{svg}</div>
                <div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                color:#484f58;letter-spacing:0.12em;text-transform:uppercase;
                                margin-bottom:4px;">{c['label']}</div>
                    <div class="metric-value" style="font-family:'Syne',sans-serif;
                                font-size:2rem;font-weight:800;color:#e6edf3;line-height:1;">
                        {c['value']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                color:{c['color']};margin-top:4px;">
                        {c['pct']}% {c['sub']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Stacked category breakdown bar ────────────────────────────────────────
    cat_counts = df["Category"].value_counts()
    cat_colors = {"Drop-off": "#f85149", "Engagement": "#58a6ff",
                  "Auth": "#d29922", "Conversion": "#3fb950"}
    segments_html = ""
    legend_html   = ""
    for cat, cnt in cat_counts.items():
        pct   = cnt / total * 100
        color = cat_colors.get(cat, "#484f58")
        segments_html += f"""
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color};
                         height:100%;position:relative;min-width:2px;"
                 title="{cat}: {cnt} ({pct:.0f}%)">
            </div>"""
        legend_html += f"""
            <div style="display:flex;align-items:center;gap:5px;">
                <div style="width:8px;height:8px;border-radius:2px;
                            background:{color};flex-shrink:0;"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                             color:#8b949e;">{cat} <span style="color:{color};">{cnt}</span></span>
            </div>"""

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;padding:16px 20px;margin-bottom:8px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
            Category Breakdown
        </div>
        <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;gap:2px;margin-bottom:12px;">
            {segments_html}
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;">{legend_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Severity mini sparkline bars ──────────────────────────────────────────
    sev_counts = df["Severity"].value_counts()
    sev_colors = {"High": "#f85149", "Medium": "#d29922", "Low": "#3fb950"}
    sev_max    = sev_counts.max() if len(sev_counts) else 1

    sev_bars = ""
    for sev in ["High", "Medium", "Low"]:
        cnt   = sev_counts.get(sev, 0)
        w     = round(cnt / sev_max * 100) if sev_max else 0
        color = sev_colors[sev]
        sev_bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:#8b949e;width:50px;text-align:right;">{sev}</div>
            <div style="flex:1;height:7px;background:#1e2d3d;border-radius:4px;overflow:hidden;">
                <div class="bar-fill" style="width:{w}%;height:100%;
                             background:{color};border-radius:4px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:{color};width:28px;">{cnt}</div>
        </div>"""

    # ── Platform mini bars ────────────────────────────────────────────────────
    plt_counts = df["Platform"].value_counts()
    plt_max    = plt_counts.max() if len(plt_counts) else 1
    plt_bars   = ""
    plt_colors = {"Android": "#3fb950", "iOS": "#58a6ff", "Web": "#d29922"}
    for plt_name in ["Android", "iOS", "Web"]:
        cnt   = plt_counts.get(plt_name, 0)
        w     = round(cnt / plt_max * 100) if plt_max else 0
        color = plt_colors.get(plt_name, "#484f58")
        plt_bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:#8b949e;width:56px;text-align:right;">{plt_name}</div>
            <div style="flex:1;height:7px;background:#1e2d3d;border-radius:4px;overflow:hidden;">
                <div class="bar-fill" style="width:{w}%;height:100%;
                             background:{color};border-radius:4px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:{color};width:28px;">{cnt}</div>
        </div>"""

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                    padding:16px 20px;height:100%;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
                Severity Split</div>
            {sev_bars}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                    padding:16px 20px;height:100%;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
                Platform Split</div>
            {plt_bars}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Top issues chip strip ─────────────────────────────────────────────────
    top_events = df["Event"].value_counts().head(4)
    chips_html = ""
    chip_colors = ["#f85149", "#d29922", "#58a6ff", "#3fb950"]
    for i, (evt, cnt) in enumerate(top_events.items()):
        color = chip_colors[i % len(chip_colors)]
        delay = i * 0.1
        chips_html += f"""
        <div class="chip-pop" style="display:inline-flex;align-items:center;gap:7px;
                    background:rgba(255,255,255,0.03);border:1px solid #1e2d3d;
                    border-radius:20px;padding:6px 12px;animation-delay:{delay}s;">
            <div style="width:6px;height:6px;border-radius:50%;background:{color};flex-shrink:0;"></div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#c9d1d9;">
                {evt[:38]}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                         color:{color};background:rgba(255,255,255,0.05);
                         border-radius:10px;padding:1px 7px;">{cnt}×</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                padding:14px 18px;margin-bottom:4px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
            🔥 Top Events</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feedback_dashboard(df):
    total    = len(df)
    critical = len(df[df["Priority"] == "Critical"])
    bugs     = len(df[df["Category"] == "Bug"])
    avg_rat  = df["Rating"].mean()

    # ── Donut metric cards ─────────────────────────────────────────────────────
    cards = [
        {"label": "Total Feedback",  "value": total,    "sub": "responses",    "color": "#58a6ff",  "pct": 100,       "display": str(total)},
        {"label": "Critical Issues", "value": critical, "sub": "of total",     "color": "#f85149",  "pct": round(critical/total*100) if total else 0, "display": str(critical)},
        {"label": "Bug Reports",     "value": bugs,     "sub": "defects filed","color": "#d29922",  "pct": round(bugs/total*100) if total else 0, "display": str(bugs)},
        {"label": "Avg Rating",      "value": round(avg_rat * 20), "sub": "user satisfaction", "color": "#3fb950", "pct": round(avg_rat*20), "display": f"{avg_rat:.1f}★"},
    ]

    cols = st.columns(4)
    for col, c in zip(cols, cards):
        svg = _donut_ring_svg(c["value"], total if c["label"] != "Avg Rating" else 100, c["color"])
        col.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:14px;
                    padding:20px 18px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,{c['color']},transparent);"></div>
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="flex-shrink:0;">{svg}</div>
                <div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                color:#484f58;letter-spacing:0.12em;text-transform:uppercase;
                                margin-bottom:4px;">{c['label']}</div>
                    <div class="metric-value" style="font-family:'Syne',sans-serif;
                                font-size:1.85rem;font-weight:800;color:#e6edf3;line-height:1;">
                        {c['display']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                                color:{c['color']};margin-top:4px;">
                        {c['pct']}% {c['sub']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Priority stacked bar ──────────────────────────────────────────────────
    pri_counts = df["Priority"].value_counts()
    pri_colors = {"Critical": "#f85149", "High": "#d29922",
                  "Medium": "#58a6ff",   "Low": "#3fb950"}
    seg_html = ""
    leg_html = ""
    for pri in ["Critical", "High", "Medium", "Low"]:
        cnt   = pri_counts.get(pri, 0)
        if cnt == 0: continue
        pct   = cnt / total * 100
        color = pri_colors[pri]
        seg_html += f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};height:100%;min-width:2px;" title="{pri}: {cnt}"></div>'
        leg_html += f"""
            <div style="display:flex;align-items:center;gap:5px;">
                <div style="width:8px;height:8px;border-radius:2px;background:{color};flex-shrink:0;"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#8b949e;">
                    {pri} <span style="color:{color};">{cnt}</span></span>
            </div>"""

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;padding:16px 20px;margin-bottom:8px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">Priority Breakdown</div>
        <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;gap:2px;margin-bottom:12px;">
            {seg_html}
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;">{leg_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Rating histogram + Category bars side by side ─────────────────────────
    rating_counts = df["Rating"].value_counts().sort_index()
    rat_max = rating_counts.max() if len(rating_counts) else 1
    rat_stars = {1: "#f85149", 2: "#d29922", 3: "#58a6ff", 4: "#58a6ff", 5: "#3fb950"}
    rat_bars = ""
    for star in [1, 2, 3, 4, 5]:
        cnt   = rating_counts.get(star, 0)
        w     = round(cnt / rat_max * 100) if rat_max else 0
        color = rat_stars[star]
        rat_bars += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#d29922;width:20px;">
                {'★'*star}</div>
            <div style="flex:1;height:8px;background:#1e2d3d;border-radius:4px;overflow:hidden;">
                <div class="bar-fill" style="width:{w}%;height:100%;background:{color};border-radius:4px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{color};width:24px;">{cnt}</div>
        </div>"""

    cat_counts_f = df["Category"].value_counts()
    cat_colors_f = {"Bug": "#f85149", "Performance": "#d29922",
                    "Feature": "#3fb950", "UX": "#58a6ff"}
    cat_max = cat_counts_f.max() if len(cat_counts_f) else 1
    cat_bars = ""
    for cat in ["Bug", "Performance", "Feature", "UX"]:
        cnt   = cat_counts_f.get(cat, 0)
        w     = round(cnt / cat_max * 100) if cat_max else 0
        color = cat_colors_f.get(cat, "#484f58")
        cat_bars += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        color:#8b949e;width:72px;text-align:right;">{cat}</div>
            <div style="flex:1;height:7px;background:#1e2d3d;border-radius:4px;overflow:hidden;">
                <div class="bar-fill" style="width:{w}%;height:100%;background:{color};border-radius:4px;"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{color};width:24px;">{cnt}</div>
        </div>"""

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                    padding:16px 20px;height:100%;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
                ★ Rating Histogram</div>
            {rat_bars}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                    padding:16px 20px;height:100%;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
                Category Split</div>
            {cat_bars}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Top feedback chip strip ───────────────────────────────────────────────
    top_fb = df["Feedback"].value_counts().head(4)
    chips_html = ""
    chip_colors = ["#f85149", "#d29922", "#58a6ff", "#3fb950"]
    for i, (fb, cnt) in enumerate(top_fb.items()):
        color = chip_colors[i % len(chip_colors)]
        delay = i * 0.1
        chips_html += f"""
        <div class="chip-pop" style="display:inline-flex;align-items:center;gap:7px;
                    background:rgba(255,255,255,0.03);border:1px solid #1e2d3d;
                    border-radius:20px;padding:6px 12px;animation-delay:{delay}s;">
            <div style="width:6px;height:6px;border-radius:50%;background:{color};flex-shrink:0;"></div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#c9d1d9;">
                {fb[:38]}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                         color:{color};background:rgba(255,255,255,0.05);
                         border-radius:10px;padding:1px 7px;">{cnt}×</span>
        </div>"""

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2d3d;border-radius:12px;
                padding:14px 18px;margin-bottom:4px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">
            🔥 Top Feedback</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataForge Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:16px;">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Rows per table**")
    n_rows = st.slider("", 10, 200, 50, step=10, label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:#1e2d3d;margin:20px 0;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                letter-spacing:0.15em;text-transform:uppercase;margin-bottom:12px;">Data Tables</div>
    """, unsafe_allow_html=True)

    show_behavior = st.checkbox("User Behavior Events", value=True)
    show_feedback  = st.checkbox("User Feedback",        value=True)

    st.markdown("""
    <div style="height:1px;background:#1e2d3d;margin:20px 0;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#484f58;
                letter-spacing:0.1em;margin-top:16px;line-height:1.8;">
        <div style="color:#30363d;">Templates</div>
        <div style="color:#58a6ff;">12 behavior patterns</div>
        <div style="color:#58a6ff;">12 feedback patterns</div>
        <div style="margin-top:8px;color:#30363d;">Dimensions</div>
        <div style="color:#3fb950;">Platform · Region · Age</div>
        <div style="color:#3fb950;">Severity · Priority</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
render_header()

df_b, df_f, generated_at, n_rows_used = load_data()
data_exists = df_b is not None or df_f is not None

# ── Generate button ────────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    generate = st.button("⚡  Generate Data", use_container_width=True, type="primary")

if generate:
    random.seed()  # re-seed every click → fresh data every time
    df_b         = generate_behavior_data(n_rows) if show_behavior else None
    df_f         = generate_feedback_data(n_rows) if show_feedback else None
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows_used  = n_rows
    save_data(df_b, df_f, generated_at, n_rows_used)
    st.session_state.pop("show_analysis", None)
    data_exists  = True
    st.rerun()

# ── Display data ───────────────────────────────────────────────────────────────
if data_exists:
    render_status_bar(generated_at, n_rows_used)

    if df_b is not None:
        render_section_heading("🖱️", "User Behavior Events", "Clickstream & interaction telemetry")
        # ── Advanced dashboard ──
        render_behavior_dashboard(df_b)
        # ── Raw table ──
        with st.expander("📋  View raw data table", expanded=False):
            st.dataframe(df_b, use_container_width=True, hide_index=True)
        st.download_button("⬇  Download Behavior CSV", df_b.to_csv(index=False).encode(), "behavior_events.csv", "text/csv")

    if df_f is not None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px;background:#1e2d3d;'></div>", unsafe_allow_html=True)
        render_section_heading("💬", "User Feedback", "Voice-of-customer signals & bug reports")
        # ── Advanced dashboard ──
        render_feedback_dashboard(df_f)
        # ── Raw table ──
        with st.expander("📋  View raw data table", expanded=False):
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.download_button("⬇  Download Feedback CSV", df_f.to_csv(index=False).encode(), "user_feedback.csv", "text/csv")

    # ── Action buttons row ─────────────────────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1e2d3d;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    btn1, btn2, btn3 = st.columns([1, 2, 1])
    with btn2:
        a_col, c_col = st.columns(2)
        with a_col:
            if st.button("🔍  Analyse Data", use_container_width=True, type="primary"):
                st.session_state["show_analysis"] = True
        with c_col:
            if st.button("🗑  Clear Data", use_container_width=True, type="secondary"):
                clear_saved_data()
                st.session_state.pop("show_analysis", None)
                st.rerun()

    # ── ML Analysis output ─────────────────────────────────────────────────────
    if st.session_state.get("show_analysis"):
        run_analysis(df_b, df_f)

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_empty_state()