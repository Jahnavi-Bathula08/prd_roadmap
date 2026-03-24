import streamlit as st
import pandas as pd
import random
import json
import os
import sqlite3
from datetime import datetime, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────

SAVE_FILE = "generated_data.json"
DB_FILE   = "datasets.db"

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

# ─── SQLite helpers ───────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS behavior_table (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            saved_at     TEXT,
            user_id      TEXT,
            timestamp    TEXT,
            event        TEXT,
            category     TEXT,
            severity     TEXT,
            platform     TEXT,
            region       TEXT,
            session_sec  INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback_table (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            saved_at     TEXT,
            user_id      TEXT,
            timestamp    TEXT,
            feedback     TEXT,
            category     TEXT,
            priority     TEXT,
            age_group    TEXT,
            platform     TEXT,
            rating       INTEGER
        )
    """)
    con.commit()
    con.close()

def dataset_name_exists(name: str) -> bool:
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM behavior_table WHERE dataset_name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    if not row:
        cur.execute("SELECT 1 FROM feedback_table WHERE dataset_name = ? LIMIT 1", (name,))
        row = cur.fetchone()
    con.close()
    return row is not None

def save_to_db(df_behavior, df_feedback, name: str):
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(DB_FILE)
    if df_behavior is not None:
        df_b = df_behavior.copy()
        df_b.insert(0, "dataset_name", name)
        df_b.insert(1, "saved_at", saved_at)
        df_b.columns = ["dataset_name", "saved_at", "user_id", "timestamp", "event", "category", "severity", "platform", "region", "session_sec"]
        df_b.to_sql("behavior_table", con, if_exists="append", index=False)
    if df_feedback is not None:
        df_f = df_feedback.copy()
        df_f.insert(0, "dataset_name", name)
        df_f.insert(1, "saved_at", saved_at)
        df_f.columns = ["dataset_name", "saved_at", "user_id", "timestamp", "feedback", "category", "priority", "age_group", "platform", "rating"]
        df_f.to_sql("feedback_table", con, if_exists="append", index=False)
    con.close()

def get_saved_dataset_names():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT dataset_name FROM behavior_table UNION SELECT DISTINCT dataset_name FROM feedback_table ORDER BY dataset_name")
    rows = [r[0] for r in cur.fetchall()]
    con.close()
    return rows

# ─── JSON persistence ─────────────────────────────────────────────────────────

def save_json(df_behavior, df_feedback, generated_at, n_rows):
    payload = {
        "generated_at": generated_at,
        "n_rows": n_rows,
        "behavior": df_behavior.to_dict(orient="records") if df_behavior is not None else None,
        "feedback": df_feedback.to_dict(orient="records") if df_feedback is not None else None,
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(payload, f)

def load_json():
    if not os.path.exists(SAVE_FILE):
        return None, None, None, None
    with open(SAVE_FILE, "r") as f:
        payload = json.load(f)
    df_b = pd.DataFrame(payload["behavior"]) if payload.get("behavior") else None
    df_f = pd.DataFrame(payload["feedback"]) if payload.get("feedback") else None
    return df_b, df_f, payload.get("generated_at"), payload.get("n_rows")

def clear_json():
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

# ─── Init ─────────────────────────────────────────────────────────────────────

init_db()

for key, default in [
    ("show_save_input", False),
    ("save_success_msg", ""),
    ("save_error_msg", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hello Pam", layout="wide", page_icon="assests/img.png")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(" Settings")
    n_rows = st.slider("Rows per table", 10, 200, 50, step=10)
    st.markdown("---")
    show_behavior = st.checkbox("User Behavior Events", value=True)
    show_feedback  = st.checkbox("User Feedback",        value=True)
    st.markdown("---")

    saved_names = get_saved_dataset_names()
    count_label = f"({len(saved_names)})" if saved_names else "(0)"
    if st.button(f" Saved Datasets {count_label}", use_container_width=True):
        st.switch_page("pages/saved_datasets.py")

# ─── Main ─────────────────────────────────────────────────────────────────────

st.title("Hello Pam")
st.markdown(
    "Product assistant for **Analyzing Data** and **Generating Insights** " 
    
   
)

df_b, df_f, generated_at, n_rows_used = load_json()
data_exists = df_b is not None or df_f is not None

# Generate button (top)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate = st.button("⚡ Generate Data", use_container_width=True)

if generate:
    df_b         = generate_behavior_data(n_rows) if show_behavior else None
    df_f         = generate_feedback_data(n_rows) if show_feedback else None
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows_used  = n_rows
    save_json(df_b, df_f, generated_at, n_rows_used)
    st.session_state.show_save_input  = False
    st.session_state.save_success_msg = ""
    st.session_state.save_error_msg   = ""
    data_exists = True
    st.rerun()

# Display
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

    st.markdown("---")

    # Save input panel
    if st.session_state.show_save_input:
        st.markdown("#### 💾 Save Dataset")
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            dataset_name = st.text_input("Name", placeholder="e.g. sprint_1_data, march_test_run …", label_visibility="collapsed")
        with sc2:
            confirm = st.button("✅ Confirm", use_container_width=True)
        if st.session_state.save_error_msg:
            st.error(st.session_state.save_error_msg)
        if confirm:
            name = dataset_name.strip()
            if not name:
                st.session_state.save_error_msg = "⚠️ Dataset name cannot be empty."
                st.rerun()
            elif dataset_name_exists(name):
                st.session_state.save_error_msg = f'❌ "{name}" already exists. Please choose a different name.'
                st.rerun()
            else:
                save_to_db(df_b, df_f, name)
                st.session_state.save_success_msg = f'✅ Saved as "{name}"'
                st.session_state.save_error_msg   = ""
                st.session_state.show_save_input  = False
                st.rerun()

    if st.session_state.save_success_msg:
        st.success(st.session_state.save_success_msg)

    # Save + Clear at bottom
    col1, col2, col3, col4, col5 = st.columns([1, 1.5, 0.3, 1.5, 1])
    with col2:
        if st.button("💾 Save", use_container_width=True, type="primary"):
            st.session_state.show_save_input  = not st.session_state.show_save_input
            st.session_state.save_success_msg = ""
            st.session_state.save_error_msg   = ""
            st.rerun()
    with col4:
        if st.button("🗑️ Clear Data", use_container_width=True, type="secondary"):
            clear_json()
            st.session_state.show_save_input  = False
            st.session_state.save_success_msg = ""
            st.session_state.save_error_msg   = ""
            st.rerun()

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Click **Generate Data** to create rule-based sample data.")