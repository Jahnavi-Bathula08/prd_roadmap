import streamlit as st
import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime, timedelta
from ml_analysis import run_analysis

# ─── Persistence ──────────────────────────────────────────────────────────────
SAVE_FILE = "generated_data.json"

# ─── Richer Templates ─────────────────────────────────────────────────────────

BEHAVIOR_TEMPLATES = [
    {"variations": ["User dropped at payment page", "User exited on payment screen", "User left during payment step"],
     "category": "Drop-off", "severity": "High", "peak_hours": [12, 13, 19, 20], "platforms": ["Android", "iOS"]},
    {"variations": ["User clicked add to cart", "User added item to cart", "User tapped add to cart button"],
     "category": "Engagement", "severity": "Low", "peak_hours": [10, 11, 14, 15, 20], "platforms": ["Android", "iOS", "Web"]},
    {"variations": ["User failed login attempt", "User couldn't log in", "User login rejected"],
     "category": "Auth", "severity": "Medium", "peak_hours": [8, 9, 18, 19], "platforms": ["Web", "iOS"]},
    {"variations": ["User abandoned signup flow", "User quit during registration", "User dropped off at signup"],
     "category": "Drop-off", "severity": "High", "peak_hours": [11, 12, 20, 21], "platforms": ["Web", "Android"]},
    {"variations": ["User viewed product details", "User opened product page", "User browsed product info"],
     "category": "Engagement", "severity": "Low", "peak_hours": list(range(9, 23)), "platforms": ["Android", "iOS", "Web"]},
    {"variations": ["User applied promo code", "User entered discount code", "User redeemed coupon"],
     "category": "Conversion", "severity": "Low", "peak_hours": [12, 13, 18, 19], "platforms": ["Web", "Android"]},
    {"variations": ["User removed item from cart", "User deleted product from cart", "User cleared cart item"],
     "category": "Drop-off", "severity": "Medium", "peak_hours": [13, 14, 20, 21], "platforms": ["iOS", "Web"]},
    {"variations": ["User completed checkout", "User placed order successfully", "User finished purchase"],
     "category": "Conversion", "severity": "Low", "peak_hours": [12, 13, 19, 20, 21], "platforms": ["Android", "iOS", "Web"]},
    {"variations": ["User searched for product", "User used search bar", "User typed in search field"],
     "category": "Engagement", "severity": "Low", "peak_hours": list(range(8, 22)), "platforms": ["Android", "iOS", "Web"]},
    {"variations": ["User session timed out", "User was auto-logged out", "User idle session expired"],
     "category": "Auth", "severity": "Medium", "peak_hours": [14, 15, 16, 22, 23], "platforms": ["Web"]},
    {"variations": ["User clicked on banner ad", "User tapped promotional banner", "User opened ad link"],
     "category": "Engagement", "severity": "Low", "peak_hours": [10, 11, 18, 19], "platforms": ["Android", "iOS"]},
    {"variations": ["User skipped onboarding", "User dismissed tutorial", "User exited intro screen"],
     "category": "Drop-off", "severity": "Medium", "peak_hours": [9, 10, 20, 21], "platforms": ["Android", "iOS"]},
]

FEEDBACK_TEMPLATES = [
    {"variations": ["Payment not working", "Unable to complete payment", "Payment page keeps failing"],
     "category": "Bug", "priority": "Critical",
     "age_bias": ["25-34", "35-44"], "rating_range": (1, 2)},
    {"variations": ["App is slow", "App takes too long to load", "Everything is lagging"],
     "category": "Performance", "priority": "High",
     "age_bias": ["18-24", "25-34"], "rating_range": (1, 3)},
    {"variations": ["Need dark mode", "Please add dark theme", "Dark mode is missing"],
     "category": "Feature", "priority": "Medium",
     "age_bias": ["18-24", "25-34"], "rating_range": (2, 4)},
    {"variations": ["Too many ads", "Ads are very annoying", "Please reduce ads"],
     "category": "UX", "priority": "Medium",
     "age_bias": ["18-24", "35-44"], "rating_range": (1, 3)},
    {"variations": ["Login keeps failing", "Can't sign in at all", "Login button not responding"],
     "category": "Bug", "priority": "Critical",
     "age_bias": ["45-54", "55+"], "rating_range": (1, 2)},
    {"variations": ["Images not loading", "Product photos won't show", "Broken images everywhere"],
     "category": "Bug", "priority": "High",
     "age_bias": ["25-34", "35-44"], "rating_range": (1, 2)},
    {"variations": ["Want push notifications", "Add order update alerts", "Need notification support"],
     "category": "Feature", "priority": "Low",
     "age_bias": ["25-34", "35-44"], "rating_range": (3, 5)},
    {"variations": ["Checkout flow is confusing", "Too many steps to buy", "Hard to complete purchase"],
     "category": "UX", "priority": "High",
     "age_bias": ["45-54", "55+"], "rating_range": (1, 3)},
    {"variations": ["Price filter not working", "Filter resets randomly", "Can't sort by price"],
     "category": "Bug", "priority": "Medium",
     "age_bias": ["25-34", "35-44"], "rating_range": (1, 3)},
    {"variations": ["Add wishlist feature", "Need a save-for-later option", "Want to bookmark products"],
     "category": "Feature", "priority": "Low",
     "age_bias": ["18-24", "25-34"], "rating_range": (3, 5)},
    {"variations": ["Search results are irrelevant", "Search is not accurate", "Wrong products showing in search"],
     "category": "Bug", "priority": "High",
     "age_bias": ["25-34", "35-44", "45-54"], "rating_range": (1, 3)},
    {"variations": ["App crashes on startup", "App force-closes randomly", "Frequent unexpected crashes"],
     "category": "Bug", "priority": "Critical",
     "age_bias": ["18-24", "25-34", "35-44"], "rating_range": (1, 1)},
]

# User personas — each has distinct behaviour signatures
PERSONAS = {
    "power_user":   {"session_range": (180, 600), "dropout_prob": 0.10, "platform_bias": ["iOS", "Web"]},
    "casual":       {"session_range": (20,  200), "dropout_prob": 0.35, "platform_bias": ["Android", "iOS"]},
    "frustrated":   {"session_range": (5,   60),  "dropout_prob": 0.70, "platform_bias": ["Web", "Android"]},
    "explorer":     {"session_range": (60,  300), "dropout_prob": 0.20, "platform_bias": ["Android", "iOS", "Web"]},
}

PLATFORMS  = ["Android", "iOS", "Web"]
REGIONS    = ["North", "South", "East", "West", "Central"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# ─── Save / Load ───────────────────────────────────────────────────────────────
def save_data(df_b, df_f, generated_at, n_rows):
    payload = {
        "generated_at": generated_at, "n_rows": n_rows,
        "behavior": df_b.to_dict(orient="records") if df_b is not None else None,
        "feedback": df_f.to_dict(orient="records") if df_f is not None else None,
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


# ─── Richer Generators ────────────────────────────────────────────────────────

def _realistic_timestamp(template, days_back=30):
    """
    Injects realistic temporal patterns:
    - Templates carry peak_hours → higher probability during those hours
    - Weekday/weekend split: more activity on weekdays for Auth/Bug,
      more on weekends for Engagement/Conversion
    """
    base = datetime.now() - timedelta(days=days_back)
    day_offset = random.randint(0, days_back)
    dt = base + timedelta(days=day_offset)
    dow = dt.weekday()  # 0=Mon … 6=Sun

    peak_hours = template.get("peak_hours", list(range(8, 22)))
    cat = template.get("category", "")

    # Weekend boost for shopping/engagement categories
    if cat in ("Engagement", "Conversion") and dow >= 5:
        if random.random() < 0.6:
            hour = random.choice(peak_hours)
        else:
            hour = random.randint(10, 22)
    # Auth/Bug issues spike on Mon morning & Thu evening
    elif cat in ("Auth", "Bug") and dow in (0, 3):
        hour = random.choice([8, 9, 18, 19] + peak_hours)
    else:
        # Weighted random: 70% from peak hours, 30% any
        if random.random() < 0.70:
            hour = random.choice(peak_hours)
        else:
            hour = random.randint(0, 23)

    minute = random.randint(0, 59)
    return dt.replace(hour=hour, minute=minute)


def generate_behavior_data(n=50):
    rows = []
    for _ in range(n):
        t       = random.choice(BEHAVIOR_TEMPLATES)
        persona_name, persona = random.choice(list(PERSONAS.items()))
        is_dropout = random.random() < persona["dropout_prob"]

        # Platform follows template bias 60% of the time, persona 40%
        if random.random() < 0.6:
            platform = random.choice(t.get("platforms", PLATFORMS))
        else:
            platform = random.choice(persona["platform_bias"])

        session = random.randint(*persona["session_range"])
        if is_dropout and t["category"] != "Drop-off":
            session = max(5, session // 3)

        ts = _realistic_timestamp(t)
        rows.append({
            "User ID":     f"U{random.randint(1000, 9999)}",
            "Timestamp":   ts.strftime("%Y-%m-%d %H:%M"),
            "Event":       random.choice(t["variations"]),
            "Category":    t["category"],
            "Severity":    t["severity"],
            "Platform":    platform,
            "Region":      random.choice(REGIONS),
            "Session (s)": session,
            "Persona":     persona_name,
        })
    return pd.DataFrame(rows)


def generate_feedback_data(n=50):
    rows = []
    for _ in range(n):
        t = random.choice(FEEDBACK_TEMPLATES)

        # Age skewed towards template bias 65% of the time
        if random.random() < 0.65 and t.get("age_bias"):
            age = random.choice(t["age_bias"])
        else:
            age = random.choice(AGE_GROUPS)

        # Rating from template range with small noise
        lo, hi = t.get("rating_range", (1, 5))
        rating  = max(1, min(5, lo + round(np.random.normal((hi - lo) / 2, 0.5))))

        # Platform: frustrated older users lean toward Web
        if age in ("45-54", "55+"):
            platform = random.choices(PLATFORMS, weights=[0.2, 0.3, 0.5])[0]
        else:
            platform = random.choice(PLATFORMS)

        ts = _realistic_timestamp({"peak_hours": [10, 11, 12, 19, 20, 21],
                                    "category": t["category"]})
        rows.append({
            "User ID":   f"U{random.randint(1000, 9999)}",
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "Feedback":  random.choice(t["variations"]),
            "Category":  t["category"],
            "Priority":  t["priority"],
            "Age Group": age,
            "Platform":  platform,
            "Rating":    rating,
        })
    return pd.DataFrame(rows)


# ─── Global Styles ────────────────────────────────────────────────────────────
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
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff !important; font-family: 'Syne', sans-serif !important; letter-spacing:.05em;
    }
    .stSlider > div > div > div > div { background: #58a6ff !important; }
    .stCheckbox label { color: #8b949e !important; font-size: .88rem !important; letter-spacing:.03em; }
    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stMetric"] {
        background: #0d1117 !important; border: 1px solid #1e2d3d !important;
        border-radius: 12px !important; padding: 20px 24px !important;
        position: relative; overflow: hidden; transition: border-color .2s ease;
    }
    [data-testid="stMetric"]:hover { border-color: #58a6ff !important; }
    [data-testid="stMetric"]::before {
        content: ''; position: absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg, #58a6ff, #3fb950);
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important; font-size: .75rem !important; font-weight: 600 !important;
        letter-spacing: .1em !important; text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important; font-size: 2rem !important; font-weight: 800 !important;
        font-family: 'Syne', sans-serif !important;
    }
    [data-testid="stDataFrame"] {
        background: #0d1117 !important; border: 1px solid #1e2d3d !important;
        border-radius: 12px !important; overflow: hidden;
    }
    .stButton > button {
        background: transparent !important; border: 1px solid #30363d !important;
        color: #8b949e !important; border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: .8rem !important;
        font-weight: 500 !important; letter-spacing: .05em !important;
        transition: all .2s ease !important; padding: 10px 20px !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important; color: #58a6ff !important;
        background: rgba(88,166,255,.06) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        border: none !important; color: #fff !important; font-size: .88rem !important;
        font-weight: 700 !important; letter-spacing: .06em !important;
        box-shadow: 0 0 20px rgba(56,139,253,.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
        box-shadow: 0 0 30px rgba(88,166,255,.45) !important; transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] { border-color: #da3633 !important; color: #da3633 !important; }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(218,54,51,.08) !important; border-color: #f85149 !important; color: #f85149 !important;
    }
    .stDownloadButton > button {
        background: rgba(46,160,67,.08) !important; border: 1px solid #2ea043 !important;
        color: #3fb950 !important; border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: .78rem !important;
        font-weight: 600 !important; letter-spacing: .04em !important; transition: all .2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(46,160,67,.15) !important; box-shadow: 0 0 12px rgba(63,185,80,.25) !important;
    }
    hr { border: none !important; border-top: 1px solid #1e2d3d !important; margin: 28px 0 !important; }
    .stAlert { border-radius: 10px !important; border: 1px solid !important;
               font-family: 'JetBrains Mono', monospace !important; font-size: .82rem !important; }
    .stSpinner > div { border-color: #58a6ff transparent transparent !important; }
    .streamlit-expanderHeader {
        background: #0d1117 !important; border: 1px solid #1e2d3d !important;
        border-radius: 10px !important; color: #c9d1d9 !important; font-family: 'Syne', sans-serif !important;
    }
    .streamlit-expanderContent {
        background: #0d1117 !important; border: 1px solid #1e2d3d !important; border-top: none !important;
    }
    [data-testid="column"] { padding: 0 8px !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div style="padding:48px 0 36px 0;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1f6feb,#3fb950);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:22px;flex-shrink:0;">📊</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;
                            color:#e6edf3;letter-spacing:-.02em;line-height:1;">Product Studio</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#58a6ff;
                            letter-spacing:.15em;text-transform:uppercase;margin-top:5px;">
                    Synthetic Data Generator · NLP Clustering · Anomaly Detection · ML Analysis Engine
                </div>
            </div>
        </div>
        <div style="width:100%;height:1px;background:linear-gradient(90deg,#1f6feb 0%,#3fb950 40%,transparent 100%);
                    opacity:.5;margin-top:24px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar(generated_at, n_rows_used):
    st.markdown(f"""
    <div style="background:rgba(63,185,80,.06);border:1px solid rgba(63,185,80,.25);border-radius:10px;
                padding:12px 20px;display:flex;align-items:center;gap:12px;margin-bottom:24px;">
        <div style="width:8px;height:8px;background:#3fb950;border-radius:50%;
                    box-shadow:0 0 8px rgba(63,185,80,.6);flex-shrink:0;"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#3fb950;letter-spacing:.04em;">
            DATASET LOADED — {n_rows_used} rows/table · Generated {generated_at}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_section_heading(icon, title, subtitle=None):
    sub_html = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#484f58;letter-spacing:.08em;text-transform:uppercase;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:32px 0 20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;
                         color:#e6edf3;letter-spacing:-.01em;">{title}</span>
        </div>
        {sub_html}
        <div style="width:40px;height:2px;background:linear-gradient(90deg,#58a6ff,transparent);
                    margin-top:10px;border-radius:1px;"></div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;border:1px dashed #1e2d3d;border-radius:16px;
                background:rgba(13,17,23,.5);margin:40px 0;">
        <div style="font-size:3rem;margin-bottom:16px;opacity:.4;">⚡</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                    color:#30363d;margin-bottom:8px;">No data generated yet</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#21262d;letter-spacing:.05em;">
            Configure settings in the sidebar, then click Generate Data
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── App ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 8px 0;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;
                    letter-spacing:.15em;text-transform:uppercase;margin-bottom:16px;">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Rows per table**")
    n_rows = st.slider("", 10, 200, 50, step=10, label_visibility="collapsed")

    st.markdown("""
    <div style="height:1px;background:#1e2d3d;margin:20px 0;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;
                letter-spacing:.15em;text-transform:uppercase;margin-bottom:12px;">Data Tables</div>
    """, unsafe_allow_html=True)
    show_behavior = st.checkbox("User Behavior Events", value=True)
    show_feedback  = st.checkbox("User Feedback",        value=True)

    st.markdown("""
    <div style="height:1px;background:#1e2d3d;margin:20px 0;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#484f58;
                letter-spacing:.1em;margin-top:16px;line-height:1.8;">
        <div style="color:#30363d;">Templates</div>
        <div style="color:#58a6ff;">12 behaviour patterns</div>
        <div style="color:#58a6ff;">12 feedback patterns</div>
        <div style="margin-top:8px;color:#30363d;">Personas</div>
        <div style="color:#3fb950;">Power · Casual · Frustrated · Explorer</div>
        <div style="margin-top:8px;color:#30363d;">Analysis Modules</div>
        <div style="color:#bc8cff;">NLP Clustering</div>
        <div style="color:#f85149;">Anomaly Detection</div>
        <div style="color:#d29922;">Model Comparison</div>
        <div style="color:#39d0d8;">Temporal Patterns</div>
        <div style="color:#58a6ff;">RF Priority Scoring</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
render_header()
df_b, df_f, generated_at, n_rows_used = load_data()
data_exists = df_b is not None or df_f is not None

col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    generate = st.button("⚡  Generate Data", use_container_width=True, type="primary")

if generate:
    df_b         = generate_behavior_data(n_rows) if show_behavior else None
    df_f         = generate_feedback_data(n_rows) if show_feedback else None
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows_used  = n_rows
    save_data(df_b, df_f, generated_at, n_rows_used)
    st.session_state.pop("show_analysis", None)
    data_exists = True
    st.rerun()

if data_exists:
    render_status_bar(generated_at, n_rows_used)

    if df_b is not None:
        render_section_heading("🖱️", "User Behavior Events", "Clickstream & interaction telemetry · Persona-aware generation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",   len(df_b))
        m2.metric("High Severity",  len(df_b[df_b["Severity"] == "High"]))
        m3.metric("Drop-offs",      len(df_b[df_b["Category"] == "Drop-off"]))
        m4.metric("Conversions",    len(df_b[df_b["Category"] == "Conversion"]))
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.dataframe(df_b, use_container_width=True, hide_index=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button("⬇  Download Behavior CSV", df_b.to_csv(index=False).encode(), "behavior_events.csv", "text/csv")

    if df_f is not None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px;background:#1e2d3d;'></div>", unsafe_allow_html=True)
        render_section_heading("💬", "User Feedback", "Voice-of-customer signals · Age-biased & rating-calibrated generation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Feedback",  len(df_f))
        m2.metric("Critical Issues", len(df_f[df_f["Priority"] == "Critical"]))
        m3.metric("Bug Reports",     len(df_f[df_f["Category"] == "Bug"]))
        m4.metric("Avg Rating",      f"{df_f['Rating'].mean():.1f} ⭐")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.download_button("⬇  Download Feedback CSV", df_f.to_csv(index=False).encode(), "user_feedback.csv", "text/csv")

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

    if st.session_state.get("show_analysis"):
        run_analysis(df_b, df_f)

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_empty_state()