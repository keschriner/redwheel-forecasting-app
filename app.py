import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# from src import apply_business_overrides, generate_all_prophet_forecasts_with_models

# 1. Page Configuration
st.set_page_config(
    page_title="Redwheel Inventory & Demand Forecasting", page_icon="🚲", layout="wide"
)

st.title("🚲 Redwheel Inventory & Demand Forecasting Dashboard")
st.markdown(
    "Interactive demand planning tool powered by Prophet & business override"
    " rules."
)


# 2. Load Data (Cached so it doesn't reload on every widget click)
@st.cache_data
def load_forecast_data():
    # Replace with your actual exported CSV path
    df = pd.read_csv("data/atomic_final_sales_forecast.csv", parse_dates=["ds"])
    return df


df_forecast = load_forecast_data()

# 3. Sidebar Controls for User Interaction
st.sidebar.header("Forecast Controls")

# Get unique SKU list for dropdown
sku_list = df_forecast["sku"].unique().tolist()
selected_sku = st.sidebar.selectbox("Select SKU to Inspect", sku_list)

# Optional interactive override slider for Q4 bump
q4_boost = (
    st.sidebar.slider("Simulate Q4 Marketing Bump (%)", 0, 30, 10) / 100.0
)

# 4. Filter Data for Selected SKU
sku_df = df_forecast[df_forecast["sku"] == selected_sku].sort_values("ds")

# Apply simulation override dynamically if desired
sku_df["display_yhat"] = sku_df["yhat"]
if q4_boost > 0:
    # Simple logic to apply bump to Q4 months (Oct-Dec)
    q4_mask = sku_df["ds"].dt.month.isin([10, 11, 12])
    sku_df.loc[q4_mask, "display_yhat"] *= 1 + q4_boost

# 5. Main Dashboard Display
col1, col2, col3 = st.columns(3)
latest_projection = (
    sku_df["display_yhat"].iloc[-1] if not sku_df.empty else 0
)
col1.metric("Next Week Projected Demand", f"{int(latest_projection)} units")
col2.metric("Active SKU", selected_sku)
col3.metric("Model Type", "Prophet + Exogenous Regressors")

st.subheader(f"2-Year Demand Trajectory & Risk Bands for {selected_sku}")

# Render Matplotlib Chart inside Streamlit
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    sku_df["ds"],
    sku_df["display_yhat"],
    label="Projected Demand",
    color="darkorange",
    lw=2,
)
ax.fill_between(
    sku_df["ds"],
    sku_df["yhat_lower"],
    sku_df["yhat_upper"],
    color="orange",
    alpha=0.2,
    label="80% Uncertainty Interval",
)
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

st.pyplot(fig)