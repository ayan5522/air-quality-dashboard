# ============================================================
#  🌍 Air Quality Prediction Dashboard
#  Streamlit app — XGBoost model + Analytics Dashboard
#  Feature order MUST match training data exactly.
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ─────────────────────────────────────────────
# Global style — keep matplotlib plots clean
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "grid.color":       "#eeeeee",
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 AQI Dashboard")
    st.markdown("---")
    st.subheader("📌 About this Project")
    st.info(
        "This dashboard predicts the **Air Quality Index (AQI)** "
        "using a trained **XGBoost Regressor** model built on "
        "India's Central Pollution Control Board (CPCB) station data.\n\n"
        "Use the **Prediction** tab to get instant AQI estimates, "
        "or explore the **Analytics Dashboard** tab for data insights."
    )
    st.markdown("---")
    st.subheader("👤 Developer")
    st.markdown("**AYAN ASIF MUNSHI**")
    st.markdown("**ABDULLAH ABDUL RASHEED KHAN**")
    st.markdown("**AQAB ZUBER MADRE**")
    st.markdown("**TANMAY RAKESH KAMBLE**")
    st.markdown("Air Quality Intelligence System")
    st.markdown("---")
    st.subheader("📊 AQI Scale Reference")
    st.markdown(
        """
| Range | Category |
|-------|----------|
| 0–50 | 🟢 Good |
| 51–100 | 🟡 Moderate |
| 101–200 | 🟠 Poor |
| 201–300 | 🔴 Very Poor |
| 301+ | 🟣 Severe |
"""
    )

# ─────────────────────────────────────────────
# Constants — exact feature order from training
# Confirmed from X.dtypes output in notebook:
# PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

# R² scores recorded from notebook training output (no retraining)
MODEL_R2_SCORES = {
    "Linear Regression": 0.7917,
    "XGBoost":           0.8898,
}

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str = "xgb_model.pkl"):
    """Load the pre-trained XGBoost model from disk."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# Load dataset for analytics (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str = "station_day.csv"):
    """Load the station_day CSV; returns None if file not found."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.fillna(df.mean(numeric_only=True))
    return df


# ─────────────────────────────────────────────
# AQI helper functions
# ─────────────────────────────────────────────
def get_aqi_category(aqi):
    """Return AQI bucket label matching training notebook logic."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    else:
        return "Severe"


def get_health_advice(category):
    """Return health recommendation for a given AQI category."""
    advice_map = {
        "Good":      "✅ Air quality is satisfactory. Enjoy outdoor activities!",
        "Moderate":  "⚠️ Sensitive individuals should take care.",
        "Poor":      "🚫 Avoid prolonged outdoor exertion.",
        "Very Poor": "😷 Limit outdoor activities. Wear a mask.",
        "Severe":    "🏠 Stay indoors and use masks. Keep windows closed.",
    }
    return advice_map.get(category, "No advice available.")


def get_aqi_color(category):
    """Return a hex colour for a category."""
    colors = {
        "Good":      "#00c853",
        "Moderate":  "#ffd600",
        "Poor":      "#ff6d00",
        "Very Poor": "#d50000",
        "Severe":    "#6a1b9a",
    }
    return colors.get(category, "#888888")


# ─────────────────────────────────────────────
# Analytics plot functions
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df):
    """Seaborn heatmap of numeric feature correlations."""
    numeric_df = df[FEATURE_COLUMNS + ["AQI"]].select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # show lower triangle only
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="#dddddd",
        annot_kws={"size": 8},
        ax=ax,
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return fig


def plot_model_comparison():
    """Horizontal bar chart comparing R2 scores of both models."""
    models = list(MODEL_R2_SCORES.keys())
    scores = list(MODEL_R2_SCORES.values())
    bar_colors = ["#FF5733", "#33C1FF"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(models, scores, color=bar_colors, height=0.45, edgecolor="white")

    # Annotate bars with percentage labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score * 100:.2f}%",
            va="center", ha="right",
            fontsize=12, fontweight="bold", color="white",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("R2 Score", fontsize=11)
    ax.set_title("Model Comparison — R2 Score", fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_aqi_distribution(df):
    """Histogram + KDE plot of the AQI column."""
    aqi_data = df["AQI"].dropna()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        aqi_data, kde=True, bins=60,
        color="#33C1FF", edgecolor="white", linewidth=0.4,
        line_kws={"linewidth": 2.5, "color": "#0057a8"},
        ax=ax,
    )

    # Shade AQI bucket regions
    bucket_bands = [
        (0,   50,  "#00c853", "Good"),
        (50,  100, "#ffd600", "Moderate"),
        (100, 200, "#ff6d00", "Poor"),
        (200, 300, "#d50000", "Very Poor"),
        (300, aqi_data.max() + 50, "#6a1b9a", "Severe"),
    ]
    for xmin, xmax, color, label in bucket_bands:
        ax.axvspan(xmin, xmax, alpha=0.08, color=color, label=label)

    ax.set_xlabel("AQI", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("AQI Distribution across All Stations", fontsize=14, fontweight="bold", pad=12)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for _, _, c, _ in bucket_bands]
    labels  = [lbl for _, _, _, lbl in bucket_bands]
    ax.legend(handles, labels, title="AQI Band", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.7)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────
model = load_model()
df    = load_data()

# ─────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────
st.title("🌍 Air Quality Prediction Dashboard")
st.markdown(
    "Predict the **Air Quality Index (AQI)** from pollutant measurements, "
    "or explore dataset analytics in the tabs below."
)
st.markdown("---")

# ─────────────────────────────────────────────
# Model availability gate
# ─────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model file not found.**  \n"
        "Please ensure `xgb_model.pkl` is in the same directory as `app.py`."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
# tab_predict, tab_analytics = st.tabs(["🔍 Prediction", "📊 Analytics Dashboard"])

# 🔥 Add a clear heading
st.markdown("##  Select Dashboard Section")

# 🔥 Custom CSS to style tabs
st.markdown("""
<style>
button[data-baseweb="tab"] {
    font-size: 80px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 10px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# 🔥 Improved tab labels
tab_predict, tab_analytics = st.tabs([
    "🔍 Prediction (Enter Data)",
    "📊 Analytics Dashboard (View Insights)"
])

# ───────────────────────────────────────────────────────────
#  TAB 1 — PREDICTION  (unchanged from original)
# ───────────────────────────────────────────────────────────
with tab_predict:

    st.subheader("🔬 Pollutant Input Parameters")
    st.markdown("Enter the measured concentration of each pollutant, then click **Predict AQI**.")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Particulate Matter")
        pm25 = st.number_input(
            "PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=60.0, step=0.1,
            help="Fine particulate matter <= 2.5 µm in diameter",
        )
        pm10 = st.number_input(
            "PM10 (µg/m³)", min_value=0.0, max_value=1000.0, value=100.0, step=0.1,
            help="Coarse particulate matter <= 10 µm in diameter",
        )
        co = st.number_input(
            "CO (mg/m³)", min_value=0.0, max_value=100.0, value=1.0, step=0.01,
            help="Carbon Monoxide concentration",
        )
        so2 = st.number_input(
            "SO2 (µg/m³)", min_value=0.0, max_value=500.0, value=15.0, step=0.1,
            help="Sulphur Dioxide concentration",
        )

    with col2:
        st.markdown("##### Nitrogen Compounds")
        no = st.number_input(
            "NO (µg/m³)", min_value=0.0, max_value=500.0, value=10.0, step=0.1,
            help="Nitric Oxide concentration",
        )
        no2 = st.number_input(
            "NO2 (µg/m³)", min_value=0.0, max_value=500.0, value=25.0, step=0.1,
            help="Nitrogen Dioxide concentration",
        )
        nox = st.number_input(
            "NOx (µg/m³)", min_value=0.0, max_value=500.0, value=30.0, step=0.1,
            help="Total Nitrogen Oxides (NO + NO2)",
        )
        nh3 = st.number_input(
            "NH3 (µg/m³)", min_value=0.0, max_value=500.0, value=15.0, step=0.1,
            help="Ammonia concentration",
        )

    with col3:
        st.markdown("##### Other Pollutants")
        o3 = st.number_input(
            "O3 (µg/m³)", min_value=0.0, max_value=500.0, value=40.0, step=0.1,
            help="Ground-level Ozone concentration",
        )
        benzene = st.number_input(
            "Benzene (µg/m³)", min_value=0.0, max_value=100.0, value=3.0, step=0.01,
            help="Benzene (C6H6) concentration",
        )
        toluene = st.number_input(
            "Toluene (µg/m³)", min_value=0.0, max_value=500.0, value=10.0, step=0.1,
            help="Toluene (C7H8) concentration",
        )
        xylene = st.number_input(
            "Xylene (µg/m³)", min_value=0.0, max_value=500.0, value=5.0, step=0.1,
            help="Xylene (C8H10) concentration",
        )

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict AQI", use_container_width=True, type="primary")

    if predict_clicked:
        # 1. Collect inputs in EXACT training feature order
        feature_values = [
            pm25, pm10, no, no2, nox,
            nh3, co, so2, o3, benzene, toluene, xylene,
        ]

        # 2. Validate
        try:
            features_array = np.array(feature_values, dtype=float).reshape(1, -1)
        except (ValueError, TypeError) as e:
            st.error(f"❌ Invalid input detected: {e}")
            st.stop()

        if np.any(np.isnan(features_array)):
            st.error("❌ One or more inputs contain NaN values. Please check your entries.")
            st.stop()

        # 3. Predict
        try:
            predicted_aqi = float(model.predict(features_array)[0])
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        predicted_aqi_rounded = round(predicted_aqi)
        category = get_aqi_category(predicted_aqi)
        advice   = get_health_advice(category)
        color    = get_aqi_color(category)

        # 4. Results display
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"""
                <div style="
                    background-color:{color}22;
                    border: 3px solid {color};
                    border-radius: 16px;
                    padding: 28px;
                    text-align: center;
                ">
                    <div style="font-size:18px; color:{color}; font-weight:600;">
                        Predicted AQI
                    </div>
                    <div style="font-size:72px; font-weight:800; color:{color}; line-height:1.1;">
                        {predicted_aqi_rounded}
                    </div>
                    <div style="font-size:22px; font-weight:700; color:{color}; margin-top:4px;">
                        {category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("#### 🏷️ Air Quality Category")
            if category == "Good":
                st.success(f"🟢 **{category}** — AQI {predicted_aqi_rounded}")
            elif category in ("Moderate", "Poor"):
                st.warning(f"🟠 **{category}** — AQI {predicted_aqi_rounded}")
            else:
                st.error(f"🔴 **{category}** — AQI {predicted_aqi_rounded}")

            st.markdown("#### 💡 Health Advice")
            if category == "Good":
                st.success(advice)
            elif category in ("Moderate", "Poor"):
                st.warning(advice)
            else:
                st.error(advice)

            st.markdown("#### 📈 AQI Level Indicator")
            bar_value = min(predicted_aqi_rounded / 500, 1.0)
            st.progress(bar_value, text=f"AQI {predicted_aqi_rounded} / 500")

        # 5. Input summary table
        st.markdown("---")
        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame({
            "Pollutant": [
                "PM2.5 (µg/m³)", "PM10 (µg/m³)",   "NO (µg/m³)",   "NO2 (µg/m³)",
                "NOx (µg/m³)",   "NH3 (µg/m³)",     "CO (mg/m³)",   "SO2 (µg/m³)",
                "O3 (µg/m³)",    "Benzene (µg/m³)", "Toluene (µg/m³)", "Xylene (µg/m³)",
            ],
            "Value": feature_values,
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────
#  TAB 2 — ANALYTICS DASHBOARD
# ───────────────────────────────────────────────────────────
with tab_analytics:

    st.subheader("📊 Analytics Dashboard")
    st.markdown(
        "Explore the training dataset visually — correlations between pollutants, "
        "model performance, and the distribution of AQI across monitoring stations."
    )

    if df is None:
        st.warning(
            "⚠️ `station_day.csv` not found in the app directory.  \n"
            "The **Correlation Heatmap** and **AQI Distribution** charts require the dataset.  \n"
            "The **Model Comparison** chart is always available (scores are from training logs)."
        )

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  Section A — Correlation Heatmap
    # ════════════════════════════════════════════════════════
    st.markdown("### 🔥 A. Feature Correlation Heatmap")
    st.markdown(
        "Shows how strongly each pollutant is linearly correlated with every other feature "
        "and with AQI. Values close to **+1** or **-1** indicate strong relationships."
    )

    if df is not None:
        with st.spinner("Rendering heatmap…"):
            fig_heatmap = plot_correlation_heatmap(df)
        st.pyplot(fig_heatmap, use_container_width=True)
        plt.close(fig_heatmap)

        # Quick insight callout
        numeric_cols = FEATURE_COLUMNS + ["AQI"]
        corr_with_aqi = (
            df[numeric_cols].corr()["AQI"]
            .drop("AQI")
            .abs()
            .sort_values(ascending=False)
        )
        top_feat = corr_with_aqi.index[0]
        top_val  = corr_with_aqi.iloc[0]
        st.info(f"💡 **Strongest predictor of AQI:** `{top_feat}` (|r| = {top_val:.2f})")
    else:
        st.error("Dataset unavailable — cannot render heatmap.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  Section B — Model Comparison Chart
    # ════════════════════════════════════════════════════════
    st.markdown("### 🏆 B. Model Comparison — R² Score")
    st.markdown(
        "R² (coefficient of determination) measures how well each model explains "
        "variance in AQI. Scores closer to **1.0** indicate a better fit."
    )

    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        with st.spinner("Rendering model comparison…"):
            fig_models = plot_model_comparison()
        st.pyplot(fig_models, use_container_width=True)
        plt.close(fig_models)

    with col_metrics:
        st.markdown("#### 📌 Score Summary")
        best_score = max(MODEL_R2_SCORES.values())
        for model_name, r2 in MODEL_R2_SCORES.items():
            icon = "🥇" if r2 == best_score else "🥈"
            delta_val = (r2 - min(MODEL_R2_SCORES.values())) * 100
            st.metric(
                label=f"{icon} {model_name}",
                value=f"{r2 * 100:.2f}%",
                delta=f"+{delta_val:.2f}% vs baseline" if r2 == best_score else None,
            )
        st.markdown("")
        improvement = (MODEL_R2_SCORES["XGBoost"] - MODEL_R2_SCORES["Linear Regression"]) * 100
        st.success(f"✅ **XGBoost** outperforms Linear Regression by **{improvement:.2f}%** in R².")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    #  Section C — AQI Distribution
    # ════════════════════════════════════════════════════════
    st.markdown("### 📉 C. AQI Distribution")
    st.markdown(
        "Histogram with KDE overlay showing the spread of AQI values across all "
        "monitoring stations. Coloured bands mark the standard AQI quality buckets."
    )

    if df is not None:
        with st.spinner("Rendering distribution plot…"):
            fig_dist = plot_aqi_distribution(df)
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

        # Summary statistics
        aqi_series = df["AQI"].dropna()
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Mean AQI",   f"{aqi_series.mean():.1f}")
        s_col2.metric("Median AQI", f"{aqi_series.median():.1f}")
        s_col3.metric("Max AQI",    f"{aqi_series.max():.1f}")
        s_col4.metric("Std Dev",    f"{aqi_series.std():.1f}")
    else:
        st.error("Dataset unavailable — cannot render AQI distribution.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌍 Air Quality Prediction Dashboard · "
    "Model: XGBoost Regressor · "
    "Data: CPCB India Station Day Records"
)
