import streamlit as st
import pandas as pd
import sqlite3

DB_FILE = "datasets.db"

# ─── DB helpers ───────────────────────────────────────────────────────────────

def get_all_datasets():
    """Return list of dicts: name, saved_at, behavior_count, feedback_count."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""
        SELECT dataset_name, saved_at, COUNT(*) as cnt
        FROM behavior_table
        GROUP BY dataset_name, saved_at
    """)
    b_rows = {r[0]: {"saved_at": r[1], "behavior_count": r[2], "feedback_count": 0} for r in cur.fetchall()}

    cur.execute("""
        SELECT dataset_name, COUNT(*) as cnt
        FROM feedback_table
        GROUP BY dataset_name
    """)
    for r in cur.fetchall():
        if r[0] in b_rows:
            b_rows[r[0]]["feedback_count"] = r[1]
        else:
            b_rows[r[0]] = {"saved_at": "—", "behavior_count": 0, "feedback_count": r[1]}

    con.close()
    return [{"name": k, **v} for k, v in sorted(b_rows.items())]

def get_behavior(name):
    con = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM behavior_table WHERE dataset_name = ?", con, params=(name,))
    con.close()
    return df

def get_feedback(name):
    con = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM feedback_table WHERE dataset_name = ?", con, params=(name,))
    con.close()
    return df

# ─── Session state defaults ───────────────────────────────────────────────────

for key, default in [
    ("view_dataset", None),
    ("analyze_dataset", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Saved Datasets", layout="wide", page_icon="🗂️")

with st.sidebar:
    if st.button("← Back to Generator", use_container_width=True):
        st.session_state.view_dataset    = None
        st.session_state.analyze_dataset = None
        st.switch_page("app.py")

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🗂️ Saved Datasets")
st.markdown("All datasets saved to SQLite. Click **View** to inspect the raw data or **Analyze** for insights.")
st.markdown("---")

datasets = get_all_datasets()

if not datasets:
    st.info("No datasets saved yet. Go back to the generator, generate some data and hit **💾 Save**.")
    st.stop()

# ─── Dataset list ─────────────────────────────────────────────────────────────

for ds in datasets:
    name  = ds["name"]
    c1, c2, c3, c4, c5 = st.columns([3, 2, 1.2, 1.2, 0.1])

    with c1:
        st.markdown(f"### 📁 `{name}`")
        st.caption(f"Saved at: {ds['saved_at']}  |  🖱️ {ds['behavior_count']} behavior rows  |  💬 {ds['feedback_count']} feedback rows")

    with c3:
        if st.button("👁️ View", key=f"view_{name}", use_container_width=True):
            st.session_state.view_dataset    = name
            st.session_state.analyze_dataset = None
            st.rerun()

    with c4:
        if st.button("📊 Analyze", key=f"analyze_{name}", use_container_width=True):
            st.session_state.analyze_dataset = name
            st.session_state.view_dataset    = None
            st.rerun()

    st.markdown("---")

# ─── VIEW panel ───────────────────────────────────────────────────────────────

if st.session_state.view_dataset:
    name = st.session_state.view_dataset
    st.markdown(f"## 👁️ Viewing — `{name}`")

    df_b = get_behavior(name)
    df_f = get_feedback(name)

    if not df_b.empty:
        st.subheader("🖱️ Behavior Table")
        st.dataframe(df_b, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Behavior CSV",
            df_b.to_csv(index=False).encode(),
            f"{name}_behavior.csv", "text/csv",
            key=f"dl_b_{name}"
        )

    if not df_f.empty:
        st.markdown("---")
        st.subheader("💬 Feedback Table")
        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Feedback CSV",
            df_f.to_csv(index=False).encode(),
            f"{name}_feedback.csv", "text/csv",
            key=f"dl_f_{name}"
        )

    if st.button("✖ Close View", key="close_view"):
        st.session_state.view_dataset = None
        st.rerun()

# ─── ANALYZE panel ────────────────────────────────────────────────────────────

if st.session_state.analyze_dataset:
    name = st.session_state.analyze_dataset
    st.markdown(f"## 📊 Analyzing — `{name}`")

    df_b = get_behavior(name)
    df_f = get_feedback(name)

    if not df_b.empty:
        st.subheader("🖱️ Behavior Insights")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",  len(df_b))
        m2.metric("High Severity", len(df_b[df_b["severity"] == "High"]))
        m3.metric("Drop-offs",     len(df_b[df_b["category"] == "Drop-off"]))
        m4.metric("Conversions",   len(df_b[df_b["category"] == "Conversion"]))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Events by Category**")
            st.bar_chart(df_b["category"].value_counts())
        with col2:
            st.markdown("**Events by Severity**")
            st.bar_chart(df_b["severity"].value_counts())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Events by Platform**")
            st.bar_chart(df_b["platform"].value_counts())
        with col2:
            st.markdown("**Events by Region**")
            st.bar_chart(df_b["region"].value_counts())

    if not df_f.empty:
        st.markdown("---")
        st.subheader("💬 Feedback Insights")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Feedback",  len(df_f))
        m2.metric("Critical Issues", len(df_f[df_f["priority"] == "Critical"]))
        m3.metric("Bug Reports",     len(df_f[df_f["category"] == "Bug"]))
        m4.metric("Avg Rating",      f"{df_f['rating'].mean():.1f} ⭐")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feedback by Category**")
            st.bar_chart(df_f["category"].value_counts())
        with col2:
            st.markdown("**Feedback by Priority**")
            st.bar_chart(df_f["priority"].value_counts())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feedback by Platform**")
            st.bar_chart(df_f["platform"].value_counts())
        with col2:
            st.markdown("**Ratings Distribution**")
            st.bar_chart(df_f["rating"].value_counts().sort_index())

    if st.button("✖ Close Analysis", key="close_analyze"):
        st.session_state.analyze_dataset = None
        st.rerun()