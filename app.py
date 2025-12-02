# app.py  –  LiPo Battery RUL Dashboard (clean version)

import pathlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="LiPo Battery RUL Dashboard",
    page_icon="https://www.flaticon.com/free-icon/battery_5998565",
    layout="wide",
)

# -------------------------------------------------
# Global style (fonts, navbar, cards…)
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font & background */
    body, .stApp {
        font-family: "Segoe UI", system-ui, sans-serif;
        background-color: #ffffff;
    }
    .main {
        padding-top: 1rem;
    }

    /* Title */
    .app-title {
        font-size: 2.3rem;
        font-weight: 750;
        color: #003366;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-subtitle {
        color: #555;
        margin-top: 0.3rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }

    /* Custom navigation pills */
    .nav-container {
        margin-bottom: 1.2rem;
    }
    .nav-pill {
        padding: 0.45rem 1.4rem;
        border-radius: 999px;
        border: 2px solid #0059b3;
        background-color: #ffffff;
        color: #0059b3;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        text-align: center;
    }
    .nav-pill:hover {
        background-color: #e6f0ff;
    }
    .nav-pill-selected {
        background-color: #0059b3 !important;
        color: #ffffff !important;
        border-color: #0059b3 !important;
    }

    /* Remove default blue focus outline around buttons */
    button:focus:not(:focus-visible) {
        outline: none;
        box-shadow: none;
    }

    /* Metric cards */
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #777;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #003366;
    }
    .metric-chip {
        display: inline-block;
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    .chip-green {
        background-color: #e6f7ec;
        color: #1c7c3b;
    }
    .chip-orange {
        background-color: #fff5e6;
        color: #c15a00;
    }

    /* Small caption under sections */
    .section-caption {
        font-size: 0.8rem;
        color: #777;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Utility: paths, caching
# -------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "lipo5_final_dataset.csv"
MODEL_PATH = BASE_DIR / "rul_random_forest.pkl"

FEATURE_COLS = ["capacity", "Avg_V", "Min_V", "Max_V", "Drop_Rate", "Avg_I", "Duration"]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Make sure Cycle is integer
    if "Cycle" in df.columns:
        df["Cycle"] = df["Cycle"].astype(int)

    # Ensure RUL columns exist (safety)
    if "RUL (cycles)" not in df.columns:
        max_cycle = df["Cycle"].max()
        df["RUL (cycles)"] = max_cycle - df["Cycle"]

    if "capacity" in df.columns and "RUL (%)" not in df.columns:
        initial_capacity = df["capacity"].iloc[0]
        df["RUL (%)"] = (df["capacity"] / initial_capacity) * 100

    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


df = load_dataset()
model = load_model()

# -------------------------------------------------
# Plot functions
# -------------------------------------------------
def plot_capacity_degradation(df_plot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_plot["Cycle"], df_plot["capacity"], label="Capacity (raw)", color="#FF8C00")
    ax.plot(
        df_plot["Cycle"],
        df_plot["capacity"].rolling(10).mean(),
        label="Smoothed Capacity (Rolling Mean)",
        color="#0059b3",
        linewidth=2,
    )
    ax.set_title("Capacity Degradation Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def plot_rul_trend(df_plot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_plot["Cycle"], df_plot["RUL (%)"], linewidth=2, color="#008000")
    ax.set_title("Remaining Useful Life (RUL) Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (%)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_drop_rate(df_plot: pd.DataFrame):
    # Filter crazy outliers to keep plot readable
    clean = df_plot[df_plot["Drop_Rate"] < 0.02]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(clean["Cycle"], clean["Drop_Rate"], color="#FF4500", linewidth=2)
    ax.set_title("Drop Rate vs Cycle", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Drop Rate (V per cycle)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_correlation_heatmap(df_plot: pd.DataFrame):
    cols = [
        "Cycle",
        "capacity",
        "Avg_V",
        "Min_V",
        "Max_V",
        "Drop_Rate",
        "Avg_I",
        "Duration",
        "RUL (%)",
        "RUL (cycles)",
    ]
    existing = [c for c in cols if c in df_plot.columns]
    corr = df_plot[existing].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(existing)))
    ax.set_xticklabels(existing, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(existing)))
    ax.set_yticklabels(existing, fontsize=8)

    # Add numbers
    for i in range(len(existing)):
        for j in range(len(existing)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    st.pyplot(fig)


def plot_pred_vs_actual_samples(df_plot: pd.DataFrame, model, n_samples: int = 10):
    sampled = df_plot.sample(n_samples, random_state=42).sort_values("Cycle")
    X = sampled[FEATURE_COLS]
    y_true = sampled["RUL (cycles)"]
    y_pred = model.predict(X)

    idx = np.arange(len(sampled))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(idx - 0.2, y_true, width=0.4, label="Actual RUL")
    ax.bar(idx + 0.2, y_pred, width=0.4, label="Predicted RUL")
    ax.set_xticks(idx)
    ax.set_xticklabels(sampled["Cycle"].astype(int))
    ax.set_xlabel("Cycle (sampled)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Predicted vs Actual RUL for Random Samples", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)


# -------------------------------------------------
# Header (title + nav)
# -------------------------------------------------
st.markdown(
    """
    <div class="app-title">
        🔋 LiPo Battery Health Dashboard
    </div>
    <div class="app-subtitle">
        Monitor capacity degradation and predict Remaining Useful Life (RUL) from stress-discharge cycles.
    </div>
    """,
    unsafe_allow_html=True,
)

# Custom navigation pills using buttons + session_state
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"

pages = ["Dashboard", "RUL Prediction", "Data Explorer"]

st.markdown('<div class="nav-container">', unsafe_allow_html=True)
nav_cols = st.columns(len(pages))

for i, page_name in enumerate(pages):
    selected = st.session_state["active_page"] == page_name
    class_name = "nav-pill nav-pill-selected" if selected else "nav-pill"
    with nav_cols[i]:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            help=page_name,
        ):
            st.session_state["active_page"] = page_name
    # Inject style on the last rendered button
    st.markdown(
        f"""
        <style>
        div[data-testid="stButton"][key="nav_{page_name}"] > button {{
            width: 100%;
        }}
        div[data-testid="stButton"][key="nav_{page_name}"] > button {{
            border-radius: 999px;
            border: 2px solid #0059b3;
            background-color: {"#0059b3" if selected else "#ffffff"};
            color: {"#ffffff" if selected else "#0059b3"};
            font-weight: 600;
        }}
        div[data-testid="stButton"][key="nav_{page_name}"] > button:hover {{
            background-color: {"#00408a" if selected else "#e6f0ff"};
            border-color: #00408a;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

current_page = st.session_state["active_page"]

# -------------------------------------------------
# PAGE 1: Dashboard
# -------------------------------------------------
if current_page == "Dashboard":
    last_row = df.loc[df["Cycle"].idxmax()]
    initial_capacity = df["capacity"].iloc[0]
    current_capacity = last_row["capacity"]
    capacity_pct = current_capacity / initial_capacity * 100

    # Predict RUL for latest cycle
    latest_features = last_row[FEATURE_COLS].values.reshape(1, -1)
    estimated_rul_cycles = float(model.predict(latest_features)[0])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-label">Current Cycle</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{int(last_row["Cycle"])}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-chip chip-green">{int(df["Cycle"].max())} total cycles so far</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown('<div class="metric-label">Current Capacity</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{current_capacity:.3f}</div>', unsafe_allow_html=True)
        delta = capacity_pct - 100
        color_class = "chip-green" if delta >= -20 else "chip-orange"
        st.markdown(
            f'<div class="metric-chip {color_class}">{capacity_pct:.1f}% of initial</div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown('<div class="metric-label">Estimated RUL (cycles)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{estimated_rul_cycles:.0f}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-chip chip-green">Model prediction from latest cycle</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Capacity Degradation Over Cycles")
        st.markdown(
            '<div class="section-caption">Smoothed trend highlights gradual health loss over time.</div>',
            unsafe_allow_html=True,
        )
        plot_capacity_degradation(df)

    with right:
        st.subheader("Remaining Useful Life (RUL) Trend")
        st.markdown(
            '<div class="section-caption">RUL (%) decreases as the battery approaches its end-of-life threshold.</div>',
            unsafe_allow_html=True,
        )
        plot_rul_trend(df)

    st.markdown("---")
    st.subheader("Additional Indicator")
    st.markdown(
        '<div class="section-caption">Drop rate shows how sharply voltage falls between cycles.</div>',
        unsafe_allow_html=True,
    )
    plot_drop_rate(df)

# -------------------------------------------------
# PAGE 2: RUL Prediction
# -------------------------------------------------
elif current_page == "RUL Prediction":
    st.subheader("Interactive RUL Prediction")

    st.markdown(
        '<div class="section-caption">Select a cycle from the dataset to inspect its features and predicted RUL.</div>',
        unsafe_allow_html=True,
    )

    min_cycle = int(df["Cycle"].min())
    max_cycle = int(df["Cycle"].max())

    cycle_selected = st.slider(
        "Select cycle number", min_value=min_cycle, max_value=max_cycle, value=min_cycle, step=1
    )

    row = df[df["Cycle"] == cycle_selected].iloc[0]
    X_sample = row[FEATURE_COLS].values.reshape(1, -1)
    rul_pred = float(model.predict(X_sample)[0])

    st.markdown("#### Cycle Data")
    st.dataframe(row.to_frame().T, use_container_width=True)

    st.markdown("#### Predicted RUL (cycles)")
    st.markdown(
        f"""
        <div class="metric-value">{rul_pred:.1f}</div>
        <div class="section-caption">This is the remaining number of cycles estimated by the model.</div>
        """,
        unsafe_allow_html=True,
    )

    # Simple visual bar (relative to max RUL)
    max_rul = df["RUL (cycles)"].max()
    progress = max(min(rul_pred / max_rul, 1.0), 0.0)
    st.progress(progress)

# -------------------------------------------------
# PAGE 3: Data Explorer
# -------------------------------------------------
elif current_page == "Data Explorer":
    st.subheader("Data Explorer")

    st.markdown(
        '<div class="section-caption">Explore raw signals, correlations, and model behaviour.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown("##### 1) Time-Series View")
        y_option = st.selectbox(
            "Select feature vs Cycle",
            ["capacity", "Avg_V", "Min_V", "Max_V", "Drop_Rate", "Avg_I", "Duration", "RUL (%)"],
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["Cycle"], df[y_option], linewidth=1.8, color="#0059b3")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(y_option)
        ax.set_title(f"{y_option} vs Cycle")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with c2:
        st.markdown("##### 2) Random Samples: Actual vs Predicted RUL")
        n_samples = st.slider("Number of random samples", 5, 20, 10, step=1)
        plot_pred_vs_actual_samples(df, model, n_samples=n_samples)

    st.markdown("---")
    st.markdown("##### 3) Feature Correlation Heatmap")
    plot_correlation_heatmap(df)

    st.markdown("##### Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)




