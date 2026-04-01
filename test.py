import streamlit as st
import pandas as pd
import random
import json
import os
import sqlite3
import base64
import hashlib
import time
from datetime import datetime, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────

SAVE_FILE       = "generated_data.json"
DB_FILE         = "datasets.db"
LOGO_VIDEO_PATH = "logo.mp4"   # ← change to your actual file path
SPLASH_DURATION = 4            # seconds to show splash

# ─── Password hashing ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ─── Database Init ────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name    TEXT NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            username     TEXT UNIQUE NOT NULL,
            password     TEXT NOT NULL,
            created_at   TEXT NOT NULL
        )
    """)

    # Behavior table
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

    # Feedback table
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

# ─── Auth helpers ─────────────────────────────────────────────────────────────

def register_user(full_name, email, username, password) -> tuple[bool, str]:
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        con.close()
        return False, "email_exists"
    cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        con.close()
        return False, "username_exists"
    cur.execute(
        "INSERT INTO users (full_name, email, username, password, created_at) VALUES (?, ?, ?, ?, ?)",
        (full_name, email, username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    con.commit()
    con.close()
    return True, "ok"

def login_user(username_or_email, password) -> tuple[bool, dict]:
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute(
        "SELECT id, full_name, email, username FROM users WHERE (username = ? OR email = ?) AND password = ?",
        (username_or_email, username_or_email, hash_password(password))
    )
    row = cur.fetchone()
    con.close()
    if row:
        return True, {"id": row[0], "full_name": row[1], "email": row[2], "username": row[3]}
    return False, {}

# ─── Dataset DB helpers ───────────────────────────────────────────────────────

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
        df_b.columns = ["dataset_name","saved_at","user_id","timestamp","event","category","severity","platform","region","session_sec"]
        df_b.to_sql("behavior_table", con, if_exists="append", index=False)
    if df_feedback is not None:
        df_f = df_feedback.copy()
        df_f.insert(0, "dataset_name", name)
        df_f.insert(1, "saved_at", saved_at)
        df_f.columns = ["dataset_name","saved_at","user_id","timestamp","feedback","category","priority","age_group","platform","rating"]
        df_f.to_sql("feedback_table", con, if_exists="append", index=False)
    con.close()

def get_saved_dataset_names():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT DISTINCT dataset_name FROM behavior_table UNION SELECT DISTINCT dataset_name FROM feedback_table ORDER BY dataset_name")
    rows = [r[0] for r in cur.fetchall()]
    con.close()
    return rows

# ─── JSON helpers ─────────────────────────────────────────────────────────────

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

# ─── Data templates ───────────────────────────────────────────────────────────

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

# ─── Init DB ──────────────────────────────────────────────────────────────────

init_db()

# ─── Session state defaults ───────────────────────────────────────────────────

for key, default in [
    ("splash_done",       False),
    ("logged_in",         False),
    ("user_info",         {}),
    ("auth_mode",         "signin"),   # "signin" | "signup"
    ("show_save_input",   False),
    ("save_success_msg",  ""),
    ("save_error_msg",    ""),
    ("auth_error",        ""),
    ("auth_success",      ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SPLASH
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.splash_done:
    st.set_page_config(page_title="Loading...", layout="centered")

    with open(LOGO_VIDEO_PATH, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>
            #MainMenu, header, footer,
            [data-testid="stSidebar"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {{
                display: none !important;
                visibility: hidden !important;
            }}
            .block-container {{
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100vw !important;
            }}
            .splash-wrapper {{
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                width: 100vw;
                background : #000;
            }}
            video.splash-video {{
                max-width: 420px;
                width: 70vw;
                border-radius: 10px;
            }}
        </style>
        <div class="splash-wrapper">
            <video class="splash-video" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        </div>
    """, unsafe_allow_html=True)

    time.sleep(SPLASH_DURATION)
    st.session_state.splash_done = True
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SIGN IN / SIGN UP
# ══════════════════════════════════════════════════════════════════════════════

elif not st.session_state.logged_in:
    st.set_page_config(page_title="Welcome", layout="centered")

    # Hide sidebar & chrome on auth page
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

            #MainMenu, header, footer,
            [data-testid="stSidebar"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {
                display: none !important;
            }

            html, body, .stApp {
                font-family: 'DM Sans', sans-serif;
            }

            .block-container {
                max-width: 480px !important;
                padding: 3rem 1.5rem 2rem !important;
            }

            /* Card */
            .auth-card {
                background: black;
                border-radius: 16px;
                padding: 2.4rem 2.2rem 2rem;
                box-shadow: 0 4px 32px rgba(0,0,0,0.09), 0 1px 4px rgba(0,0,0,0.05);
            }

            .auth-title {
                font-family: 'DM Serif Display', serif;
                font-size: 2rem;
                color: white;
                margin: 0 0 0.2rem;
                line-height: 1.2;
            }

            .auth-subtitle {
                font-size: 0.88rem;
                color: #777;
                margin: 0 0 1.8rem;
                font-weight: 300;
            }

            /* Tab switcher */
            .tab-row {
                display: flex;
                gap: 0;
                background: #f4f4f5;
                border-radius: 10px;
                padding: 4px;
                margin-bottom: 1.6rem;
            }
            .tab-btn {
                flex: 1;
                padding: 0.5rem;
                border: none;
                border-radius: 7px;
                font-size: 0.88rem;
                font-family: 'DM Sans', sans-serif;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.18s ease;
                background: transparent;
                color: #888;
            }
            .tab-btn.active {
                background: white;
                color: #111;
                box-shadow: 0 1px 6px rgba(0,0,0,0.1);
            }

            /* Streamlit input overrides */
            div[data-testid="stTextInput"] label {
                font-size: 0.8rem !important;
                font-weight: 500 !important;
                color: #444 !important;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            div[data-testid="stTextInput"] input {
                border-radius: 9px !important;
                border: 1.5px solid #e5e5e5 !important;
                font-family: 'DM Sans', sans-serif !important;
                font-size: 0.95rem !important;
                padding: 0.55rem 0.8rem !important;
                transition: border 0.15s;
            }
            div[data-testid="stTextInput"] input:focus {
                border-color: #4f46e5 !important;
                box-shadow: 0 0 0 3px rgba(79,70,229,0.08) !important;
            }

            /* Primary button */
            div[data-testid="stButton"] > button[kind="primary"] {
                background: #4f46e5 !important;
                border: none !important;
                border-radius: 10px !important;
                font-family: 'DM Sans', sans-serif !important;
                font-size: 0.95rem !important;
                font-weight: 500 !important;
                padding: 0.65rem 1rem !important;
                letter-spacing: 0.02em;
                transition: background 0.2s, transform 0.1s;
            }
            div[data-testid="stButton"] > button[kind="primary"]:hover {
                background: #4338ca !important;
                transform: translateY(-1px);
            }

            /* Secondary / ghost */
            div[data-testid="stButton"] > button[kind="secondary"] {
                background: transparent !important;
                border: 1.5px solid #e5e5e5 !important;
                border-radius: 10px !important;
                font-family: 'DM Sans', sans-serif !important;
                color: #555 !important;
                font-size: 0.88rem !important;
                padding: 0.5rem 1rem !important;
            }

            .divider-text {
                text-align: center;
                font-size: 0.78rem;
                color: #bbb;
                margin: 0.6rem 0;
                position: relative;
            }

            .footer-note {
                text-align: center;
                font-size: 0.78rem;
                color: #aaa;
                margin-top: 1.4rem;
            }
        </style>
    """, unsafe_allow_html=True)

    mode = st.session_state.auth_mode

    # ── Tab switcher (HTML buttons trigger rerun via query param trick) ────────
    st.markdown(f"""
        <div class="auth-card">
            <p class="auth-title">{"Welcome back" if mode == "signin" else "Create account"}</p>
            <p class="auth-subtitle">{"Sign in to continue to your workspace." if mode == "signin" else "Join and start generating data instantly."}</p>
            <div class="tab-row">
                <button class="tab-btn {'active' if mode == 'signin' else ''}"
                    onclick="window.location.href='?mode=signin'">Sign In</button>
                <button class="tab-btn {'active' if mode == 'signup' else ''}"
                    onclick="window.location.href='?mode=signup'">Sign Up</button>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Read query param to switch mode
    params = st.query_params
    if "mode" in params and params["mode"] in ("signin", "signup"):
        if params["mode"] != st.session_state.auth_mode:
            st.session_state.auth_mode  = params["mode"]
            st.session_state.auth_error = ""
            st.session_state.auth_success = ""
            st.rerun()

    # ── Error / success banners ───────────────────────────────────────────────
    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)
    if st.session_state.auth_success:
        st.success(st.session_state.auth_success)

    # ── SIGN IN form ──────────────────────────────────────────────────────────
    if mode == "signin":
        identifier = st.text_input("Username or Email", placeholder="you@example.com or username")
        password   = st.text_input("Password", type="password", placeholder="••••••••")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if st.button("Sign In →", use_container_width=True, type="primary"):
            if not identifier or not password:
                st.session_state.auth_error = "⚠️ Please fill in all fields."
                st.rerun()
            else:
                ok, user = login_user(identifier, password)
                if ok:
                    st.session_state.logged_in  = True
                    st.session_state.user_info  = user
                    st.session_state.auth_error = ""
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.session_state.auth_error = "❌ Invalid credentials. Please try again."
                    st.rerun()

        st.markdown("""
            <p class="footer-note">Don't have an account?
                <a href="?mode=signup" style="color:#4f46e5;text-decoration:none;font-weight:500">Sign up</a>
            </p>
        """, unsafe_allow_html=True)

    # ── SIGN UP form ──────────────────────────────────────────────────────────
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            full_name = st.text_input("Full Name", placeholder="Aarav Sharma")
        with col_b:
            username = st.text_input("Username", placeholder="aarav99")

        email = st.text_input("Email Address", placeholder="aarav@example.com")

        col_c, col_d = st.columns(2)
        with col_c:
            password  = st.text_input("Password", type="password", placeholder="Min 6 chars")
        with col_d:
            confirm_pw = st.text_input("Confirm Password", type="password", placeholder="Repeat password")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if st.button("Create Account →", use_container_width=True, type="primary"):
            # Validation
            if not all([full_name, username, email, password, confirm_pw]):
                st.session_state.auth_error = "⚠️ Please fill in all fields."
                st.rerun()
            elif len(password) < 6:
                st.session_state.auth_error = "⚠️ Password must be at least 6 characters."
                st.rerun()
            elif password != confirm_pw:
                st.session_state.auth_error = "⚠️ Passwords do not match."
                st.rerun()
            elif "@" not in email or "." not in email:
                st.session_state.auth_error = "⚠️ Please enter a valid email address."
                st.rerun()
            else:
                ok, msg = register_user(full_name.strip(), email.strip().lower(), username.strip(), password)
                if ok:
                    st.session_state.auth_success = f"✅ Account created! Welcome, {full_name.split()[0]}. Please sign in."
                    st.session_state.auth_error   = ""
                    st.session_state.auth_mode    = "signin"
                    st.query_params.clear()
                    st.rerun()
                elif msg == "email_exists":
                    st.session_state.auth_error = "❌ This email is already registered."
                    st.rerun()
                elif msg == "username_exists":
                    st.session_state.auth_error = f'❌ Username "{username}" is taken. Try another.'
                    st.rerun()

        st.markdown("""
            <p class="footer-note">Already have an account?
                <a href="?mode=signin" style="color:#4f46e5;text-decoration:none;font-weight:500">Sign in</a>
            </p>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.set_page_config(page_title="Rule-Based Data Generator", layout="wide", page_icon="📊")

    # ─── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        user = st.session_state.user_info
        st.markdown(f"""
            <div style="padding:0.8rem 0 1rem;">
                <div style="font-size:0.75rem;color:#999;text-transform:uppercase;letter-spacing:0.05em">Signed in as</div>
                <div style="font-size:1rem;font-weight:600;color:#111;margin-top:2px">{user.get('full_name','')}</div>
                <div style="font-size:0.78rem;color:#aaa">{user.get('email','')}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        st.header("⚙️ Settings")
        n_rows = st.slider("Rows per table", 10, 200, 50, step=10)
        st.markdown("---")
        show_behavior = st.checkbox("User Behavior Events", value=True)
        show_feedback  = st.checkbox("User Feedback",        value=True)
        st.markdown("---")

        saved_names = get_saved_dataset_names()
        count_label = f"({len(saved_names)})" if saved_names else "(0)"
        if st.button(f"🗂️ Saved Datasets {count_label}", use_container_width=True):
            st.switch_page("pages/saved_datasets.py")

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            for k in ["logged_in", "user_info", "splash_done",
                      "show_save_input", "save_success_msg", "save_error_msg"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # ─── Main content ─────────────────────────────────────────────────────────
    user = st.session_state.user_info
    st.title("📊 Rule-Based Data Generator")
    st.markdown(
        f"Hello, **{user.get('full_name', 'there')}** 👋 — Generates realistic **User Behavior Events** and **User Feedback** — "
        "rule-based topics, random combinations every click. "
        "**Data persists across refreshes** until cleared."
    )

    df_b, df_f, generated_at, n_rows_used = load_json()
    data_exists = df_b is not None or df_f is not None

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
            with st.expander("📈 Category Breakdown", expanded=True):
                st.bar_chart(df_b["Category"].value_counts())
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
            with st.expander("📈 Priority Breakdown", expanded=True):
                st.bar_chart(df_f["Priority"].value_counts())
            st.dataframe(df_f, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Feedback CSV", df_f.to_csv(index=False).encode(), "user_feedback.csv", "text/csv")

        st.markdown("---")

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