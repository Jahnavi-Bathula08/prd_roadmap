import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

# ─── Rule-based Templates with Variations ────────────────────────────────────

BEHAVIOR_TEMPLATES = [
    {
        "variations": [
            "User dropped at payment page",
            "User exited on payment screen",
            "User left during payment step",
        ],
        "category": "Drop-off", "severity": "High",
    },
    {
        "variations": [
            "User clicked add to cart",
            "User added item to cart",
            "User tapped add to cart button",
        ],
        "category": "Engagement", "severity": "Low",
    },
    {
        "variations": [
            "User failed login attempt",
            "User couldn't log in",
            "User login rejected",
        ],
        "category": "Auth", "severity": "Medium",
    },
    {
        "variations": [
            "User abandoned signup flow",
            "User quit during registration",
            "User dropped off at signup",
        ],
        "category": "Drop-off", "severity": "High",
    },
    {
        "variations": [
            "User viewed product details",
            "User opened product page",
            "User browsed product info",
        ],
        "category": "Engagement", "severity": "Low",
    },
    {
        "variations": [
            "User applied promo code",
            "User entered discount code",
            "User redeemed coupon",
        ],
        "category": "Conversion", "severity": "Low",
    },
    {
        "variations": [
            "User removed item from cart",
            "User deleted product from cart",
            "User cleared cart item",
        ],
        "category": "Drop-off", "severity": "Medium",
    },
    {
        "variations": [
            "User completed checkout",
            "User placed order successfully",
            "User finished purchase",
        ],
        "category": "Conversion", "severity": "Low",
    },
    {
        "variations": [
            "User searched for product",
            "User used search bar",
            "User typed in search field",
        ],
        "category": "Engagement", "severity": "Low",
    },
    {
        "variations": [
            "User session timed out",
            "User was auto-logged out",
            "User idle session expired",
        ],
        "category": "Auth", "severity": "Medium",
    },
    {
        "variations": [
            "User clicked on banner ad",
            "User tapped promotional banner",
            "User opened ad link",
        ],
        "category": "Engagement", "severity": "Low",
    },
    {
        "variations": [
            "User skipped onboarding",
            "User dismissed tutorial",
            "User exited intro screen",
        ],
        "category": "Drop-off", "severity": "Medium",
    },
]

FEEDBACK_TEMPLATES = [
    {
        "variations": [
            "Payment not working",
            "Unable to complete payment",
            "Payment page keeps failing",
        ],
        "category": "Bug", "priority": "Critical",
    },
    {
        "variations": [
            "App is slow",
            "App takes too long to load",
            "Everything is lagging",
        ],
        "category": "Performance", "priority": "High",
    },
    {
        "variations": [
            "Need dark mode",
            "Please add dark theme",
            "Dark mode is missing",
        ],
        "category": "Feature", "priority": "Medium",
    },
    {
        "variations": [
            "Too many ads",
            "Ads are very annoying",
            "Please reduce ads",
        ],
        "category": "UX", "priority": "Medium",
    },
    {
        "variations": [
            "Login keeps failing",
            "Can't sign in at all",
            "Login button not responding",
        ],
        "category": "Bug", "priority": "Critical",
    },
    {
        "variations": [
            "Images not loading",
            "Product photos won't show",
            "Broken images everywhere",
        ],
        "category": "Bug", "priority": "High",
    },
    {
        "variations": [
            "Want push notifications",
            "Add order update alerts",
            "Need notification support",
        ],
        "category": "Feature", "priority": "Low",
    },
    {
        "variations": [
            "Checkout flow is confusing",
            "Too many steps to buy",
            "Hard to complete purchase",
        ],
        "category": "UX", "priority": "High",
    },
    {
        "variations": [
            "Price filter not working",
            "Filter resets randomly",
            "Can't sort by price",
        ],
        "category": "Bug", "priority": "Medium",
    },
    {
        "variations": [
            "Add wishlist feature",
            "Need a save-for-later option",
            "Want to bookmark products",
        ],
        "category": "Feature", "priority": "Low",
    },
    {
        "variations": [
            "Search results are irrelevant",
            "Search is not accurate",
            "Wrong products showing in search",
        ],
        "category": "Bug", "priority": "High",
    },
    {
        "variations": [
            "App crashes on startup",
            "App force-closes randomly",
            "Frequent unexpected crashes",
        ],
        "category": "Bug", "priority": "Critical",
    },
]

PLATFORMS  = ["Android", "iOS", "Web"]
REGIONS    = ["North", "South", "East", "West", "Central"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

def random_timestamp(days_back=30):
    base = datetime.now() - timedelta(days=days_back)
    return base + timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

def generate_behavior_data(n=50):
    rows = []
    for _ in range(n):
        t = random.choice(BEHAVIOR_TEMPLATES)
        rows.append({
            "User ID":     f"U{random.randint(1000, 9999)}",
            "Timestamp":   random_timestamp().strftime("%Y-%m-%d %H:%M"),
            "Event":       random.choice(t["variations"]),   # ← random variation each time
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
            "Feedback":  random.choice(t["variations"]),     # ← random variation each time
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
    "rule-based topics, **random combinations every click**."
)

with st.sidebar:
    st.header("⚙️ Settings")
    n_rows = st.slider("Rows per table", 10, 200, 50, step=10)
    st.markdown("---")
    show_behavior = st.checkbox("User Behavior Events", value=True)
    show_feedback  = st.checkbox("User Feedback",        value=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate = st.button("⚡ Generate Data", use_container_width=True)

if generate:
    st.success(f"✅ Generated {n_rows} rows each — unique every click!")
    st.markdown("---")

    if show_behavior:
        st.subheader("🖱️ User Behavior Events")
        df_b = generate_behavior_data(n_rows)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",  len(df_b))
        m2.metric("High Severity", len(df_b[df_b["Severity"] == "High"]))
        m3.metric("Drop-offs",     len(df_b[df_b["Category"] == "Drop-off"]))
        m4.metric("Conversions",   len(df_b[df_b["Category"] == "Conversion"]))

        with st.expander("📈 Category Breakdown", expanded=True):
            st.bar_chart(df_b["Category"].value_counts())

        st.dataframe(df_b, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV", df_b.to_csv(index=False).encode(),
            "behavior_events.csv", "text/csv"
        )

    if show_feedback:
        st.markdown("---")
        st.subheader("💬 User Feedback")
        df_f = generate_feedback_data(n_rows)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Feedback",  len(df_f))
        m2.metric("Critical Issues", len(df_f[df_f["Priority"] == "Critical"]))
        m3.metric("Bug Reports",     len(df_f[df_f["Category"] == "Bug"]))
        m4.metric("Avg Rating",      f"{df_f['Rating'].mean():.1f} ⭐")

        with st.expander("📈 Priority Breakdown", expanded=True):
            st.bar_chart(df_f["Priority"].value_counts())

        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV", df_f.to_csv(index=False).encode(),
            "user_feedback.csv", "text/csv"
        )

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Click **Generate Data** to create rule-based sample data.")