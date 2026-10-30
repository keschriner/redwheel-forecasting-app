import pandas as pd
import numpy as np
import logging
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

from src.config import PROCESSED_SALES_FILE, FORECAST_OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned weekly master sales dataset."""
    if not PROCESSED_SALES_FILE.exists():
        raise FileNotFoundError(f"Processed sales file not found at {PROCESSED_SALES_FILE}. Run the 'ingest' command first.")
    
    logging.info(f"Loading processed data from {PROCESSED_SALES_FILE}")
    df = pd.read_csv(PROCESSED_SALES_FILE)
    
    # Use week_start_date (matching the updated ETL output schema)
    date_col = 'week_start_date' if 'week_start_date' in df.columns else 'week_start'
    df[date_col] = pd.to_datetime(df[date_col])
    if date_col != 'week_start':
        df = df.rename(columns={date_col: 'week_start'})
        
    return df


def run_seasonal_decomposition(df: pd.DataFrame, target_col: str = 'units_sold') -> dict:
    """
    Performs seasonal decomposition using statsmodels on weekly sales data.
    Can be run globally or per channel.
    """
    logging.info("Running seasonal-trend decomposition...")
    
    # Ensure regular weekly frequency for time-series decomposition
    ts_df = df.set_index('week_start')[target_col].resample('W').sum().fillna(0)
    
    # Perform additive decomposition (assuming weekly frequency, period=52 for yearly seasonality)
    if len(ts_df) >= 104:  # At least 2 years of data
        period = 52
    else:
        period = min(12, len(ts_df) // 2)  # Fallback for shorter series

    if period < 2:
        logging.warning("Not enough data points for robust seasonal decomposition.")
        return {}

    decomposition = seasonal_decompose(ts_df, model='additive', period=period)
    
    decomp_df = pd.DataFrame({
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    }).reset_index()

    output_path = FORECAST_OUTPUTS_DIR / "seasonal_decomposition_results.csv"
    decomp_df.to_csv(output_path, index=False)
    logging.info(f"Saved decomposition results to {output_path}")
    
    return {
        "decomposition_dataframe": decomp_df,
        "model_object": decomposition
    }


def fit_prophet_model(df: pd.DataFrame, target_col: str = 'units_sold', periods_to_forecast: int = 104) -> pd.DataFrame:
    """
    Fits a Prophet model on weekly sales data and generates future forecasts (default 104 weeks / 2 years).
    """
    logging.info("Training Prophet forecasting model...")
    
    # Prepare dataframe for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df.groupby('week_start', as_index=False)[target_col].sum()
    prophet_df = prophet_df.rename(columns={'week_start': 'ds', target_col: 'y'})

    # Initialize and fit Prophet model
    model = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)

    # Create future dataframe for predictions (104 weeks out)
    future = model.make_future_dataframe(periods=periods_to_forecast, freq='W')
    forecast = model.predict(future)

    output_path = FORECAST_OUTPUTS_DIR / "prophet_forecast_results.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_path, index=False)
    logging.info(f"Saved Prophet forecast results to {output_path}")

    return forecast


def run_forecasting_pipeline() -> dict:
    """
    Executes the full forecasting and decomposition pipeline.
    """
    logging.info("Starting forecasting pipeline...")
    df = load_processed_data()

    # Identify target metric column dynamically
    target_col = 'units_sold' if 'units_sold' in df.columns else 'quantity'
    if target_col not in df.columns:
        target_col = 'gross_revenue' if 'gross_revenue' in df.columns else df.columns[-1]

    decomposition_results = run_seasonal_decomposition(df, target_col=target_col)
    forecast_results = fit_prophet_model(df, target_col=target_col)

    logging.info("Forecasting pipeline completed successfully.")
    return {
        "decomposition": decomposition_results,
        "forecast": forecast_results
    }


if __name__ == "__main__":
    run_forecasting_pipeline()