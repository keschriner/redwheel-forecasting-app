import os
from pathlib import Path
import pandas as pd

# Define base project root directory (relative to src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Directory Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
FORECAST_OUTPUTS_DIR = OUTPUTS_DIR / "forecasts"

# Ensure directories exist upon import/execution
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FORECAST_OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- File Names ---
# Raw files provided by Atomic (update filenames if necessary)
RAW_FILES = {
    "dtc": "dtc_web_orders.csv",
    "dealers": "dealer_orders_export.csv",
    "pos": "Commercial_PO_Log.csv",
    "promos": "promos.csv"
}

# Processed master artifact
PROCESSED_SALES_FILE = PROCESSED_DATA_DIR / "fact_weekly_sales.csv"

# --- Global Pandas Display Settings ---
def configure_pandas():
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 1000)