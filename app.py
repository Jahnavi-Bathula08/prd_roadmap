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

# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Rule-Based Data Generator", layout="wide")

st.title("📊 Rule-Based Data Generator")
st.markdown(
    "Generates realistic **User Behavior Events** and **User Feedback** — "
    "rule-based topics, random combinations every click. "
    "**Data persists even after page refresh** until you clear it."
)

with st.sidebar:
    st.header("⚙️ Settings")
    n_rows = st.slider("Rows per table", 10, 200, 50, step=10)
    st.markdown("---")
    show_behavior = st.checkbox("User Behavior Events", value=True)
    show_feedback  = st.checkbox("User Feedback",        value=True)

# Load from disk on every page load / refresh
df_b, df_f, generated_at, n_rows_used = load_data()
data_exists = df_b is not None or df_f is not None

# ─── Generate button ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate = st.button("⚡ Generate Data", use_container_width=True)

if generate:
    df_b         = generate_behavior_data(n_rows) if show_behavior else None
    df_f         = generate_feedback_data(n_rows) if show_feedback else None
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows_used  = n_rows
    save_data(df_b, df_f, generated_at, n_rows_used)
    # reset any previous analysis when new data is generated
    st.session_state.pop("show_analysis", None)
    data_exists  = True
    st.rerun()

# ─── Display data ─────────────────────────────────────────────────────────────
if data_exists:
    st.success(f"✅ Showing {n_rows_used} rows each — generated at {generated_at}")
    st.markdown("---")

    if df_b is not None:
        st.subheader("🖱️ User Behavior Events")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",  len(df_b))
        m2.metric("High Severity", len(df_b[df_b["Severity"] == "High"]))
        m3.metric("Drop-offs",     len(df_b[df_b["Category"] == "Drop-off"]))
        m4.metric("Conversions",   len(df_b[df_b["Category"] == "Conversion"]))
        st.dataframe(df_b, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Behavior CSV", df_b.to_csv(index=False).encode(), "behavior_events.csv", "text/csv")

    if df_f is not None:
        st.markdown("---")
        st.subheader("💬 User Feedback")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Feedback",  len(df_f))
        m2.metric("Critical Issues", len(df_f[df_f["Priority"] == "Critical"]))
        m3.metric("Bug Reports",     len(df_f[df_f["Category"] == "Bug"]))
        m4.metric("Avg Rating",      f"{df_f['Rating'].mean():.1f} ⭐")
        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Feedback CSV", df_f.to_csv(index=False).encode(), "user_feedback.csv", "text/csv")

    # ─── Action buttons row ───────────────────────────────────────────────────
    st.markdown("---")
    btn1, btn2, btn3 = st.columns([1, 2, 1])

    with btn2:
        # Analyse button (primary) + Clear button (secondary) side by side
        a_col, c_col = st.columns(2)
        with a_col:
            if st.button("🔍 Analyse Data", use_container_width=True, type="primary"):
                st.session_state["show_analysis"] = True
        with c_col:
            if st.button("🗑️ Clear Data", use_container_width=True, type="secondary"):
                clear_saved_data()
                st.session_state.pop("show_analysis", None)
                st.rerun()

    # ─── ML Analysis output (persists until new Generate or Clear) ────────────
    if st.session_state.get("show_analysis"):
        run_analysis(df_b, df_f)

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Click **Generate Data** to create rule-based sample data.")