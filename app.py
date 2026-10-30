import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Redwheel Bicycle Sales & Forecasting",
    page_icon="🚲",
    layout="wide"
)

# Define paths
PROCESSED_DATA_PATH = Path("data/processed/fact_weekly_sales.csv")
FORECAST_DATA_PATH = Path("outputs/forecasts/prophet_forecast_results.csv")
DECOMP_DATA_PATH = Path("outputs/forecasts/seasonal_decomposition_results.csv")

@st.cache_data
def load_data():
    """Load cached processed and forecasted artifacts."""
    sales_df = pd.read_csv(PROCESSED_DATA_PATH) if PROCESSED_DATA_PATH.exists() else pd.DataFrame()
    if not sales_df.empty:
        sales_df['week_start_date'] = pd.to_datetime(sales_df['week_start_date'])
        
    forecast_df = pd.read_csv(FORECAST_DATA_PATH) if FORECAST_DATA_PATH.exists() else pd.DataFrame()
    if not forecast_df.empty:
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        
    decomp_df = pd.read_csv(DECOMP_DATA_PATH) if DECOMP_DATA_PATH.exists() else pd.DataFrame()
    if not decomp_df.empty and 'week_start' in decomp_df.columns:
        decomp_df['week_start'] = pd.to_datetime(decomp_df['week_start'])
        
    return sales_df, forecast_df, decomp_df

sales_df, forecast_df, decomp_df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Controls")

if not sales_df.empty:
    channels = sales_df['channel'].unique().tolist()
    selected_channel = st.sidebar.selectbox("Select Channel", ["All Channels"] + channels)
    
    if selected_channel != "All Channels":
        filtered_sales = sales_df[sales_df['channel'] == selected_channel]
    else:
        filtered_sales = sales_df
else:
    filtered_sales = pd.DataFrame()

# --- Main Dashboard Layout ---
st.title("🚲 Redwheel Bicycle Sales & Forecasting Dashboard")
st.markdown("Interactive executive dashboard for tracking weekly sales, promotional impacts, and Prophet forecasting projections.")

if sales_df.empty:
    st.warning("No processed sales data found. Please run `python main.py ingest` in your terminal first.")
else:
    # Top-level KPIs
    total_units = filtered_sales['units_sold'].sum()
    total_revenue = filtered_sales['gross_revenue'].sum()
    avg_price = filtered_sales['avg_selling_price'].mean() if not filtered_sales.empty else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Units Sold", f"{total_units:,.0f}")
    col2.metric("Total Gross Revenue", f"${total_revenue:,.2f}")
    col3.metric("Average Selling Price", f"${avg_price:,.2f}")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Sales & Forecasts", "🔍 Seasonal Decomposition", "📋 Raw Data Explorer"])

    # with tab1:
    #     st.subheader("Historical Sales & Prophet Forecast Projections")
        
    #     if not forecast_df.empty:
    #         # Aggregate historical weekly units
    #         hist_agg = filtered_sales.groupby('week_start_date', as_index=False)['units_sold'].sum()
            
    #         fig = px.line(hist_agg, x='week_start_date', y='units_sold', title="Weekly Units Sold vs. Prophet Forecast")
    #         fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='orange', dash='dash'))
    #         fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='lightgray', width=0), showlegend=False)
    #         fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='lightgray', width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.2)', showlegend=False)
            
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.info("No forecast results found. Run `python main.py forecast` to generate Prophet projections.")

    with tab1:
        st.subheader("Historical Sales & Prophet Forecast Projections")
        
        if not forecast_df.empty:
            import plotly.graph_objects as go

            # Aggregate historical weekly units
            hist_agg = filtered_sales.groupby('week_start_date', as_index=False)['units_sold'].sum()

            fig = go.Figure()

            # 1. Add Uncertainty Interval (Shaded Region) first so it sits underneath lines
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='Forecast Uncertainty (80% CI)'
            ))

            # 2. Add Prophet Forecast Line
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name='Prophet Forecast',
                line=dict(color='orange', dash='dash')
            ))

            # 3. Add Historical Observed Sales Line
            fig.add_trace(go.Scatter(
                x=hist_agg['week_start_date'],
                y=hist_agg['units_sold'],
                mode='lines',
                name='Historical Units Sold',
                line=dict(color='#1f77b4', width=2)
            ))

            fig.update_layout(
                title="Weekly Units Sold vs. Prophet Forecast (Including Future Projections)",
                xaxis_title="Week Start Date",
                yaxis_title="Units Sold",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecast results found. Run `python main.py forecast` to generate Prophet projections.")

    with tab2:
        st.subheader("Statsmodels Seasonal Decomposition")
        if not decomp_df.empty:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # Create a 4-row vertical subplot grid sharing the x-axis
            fig_decomp = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
            )

            # Add each component trace to its respective row
            components = [
                ('observed', 1),
                ('trend', 2),
                ('seasonal', 3),
                ('residual', 4)
            ]

            for col_name, row_num in components:
                if col_name in decomp_df.columns:
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=decomp_df['week_start'], 
                            y=decomp_df['observed'] if col_name == 'observed' else decomp_df[col_name], 
                            mode='lines', 
                            name=col_name.capitalize()
                        ),
                        row=row_num, col=1
                    )

            fig_decomp.update_layout(
                height=800, 
                title_text="Weekly Sales Seasonal Decomposition Breakdown",
                showlegend=False
            )
            
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.info("No decomposition results found. Run `python main.py forecast` to generate decomposition data.")

    with tab3:
        st.subheader("Processed Master Weekly Sales Table")
        st.dataframe(filtered_sales, use_container_width=True)