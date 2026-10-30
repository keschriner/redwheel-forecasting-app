import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.config import RAW_DATA_DIR, RAW_FILES, PROCESSED_SALES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load raw CSV datasets from the raw data directory."""
    raw_data = {}
    for key, filename in RAW_FILES.items():
        file_path = RAW_DATA_DIR / filename
        if not file_path.exists():
            logging.warning(f"File not found: {file_path}. Returning empty DataFrame placeholder.")
            raw_data[key] = pd.DataFrame()
        else:
            logging.info(f"Loading raw file: {filename}")
            raw_data[key] = pd.read_csv(file_path)
    return raw_data


def apply_promotions(combined_df: pd.DataFrame, promos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized and optimized application of promotional periods to transactions.
    """
    combined_df['is_promo_active'] = False
    combined_df['promo_discount'] = 0.0

    if promos_df.empty or 'Start' not in promos_df.columns or 'End' not in promos_df.columns:
        return combined_df

    # Ensure order_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(combined_df['order_date']):
        combined_df['order_date'] = pd.to_datetime(combined_df['order_date'])

    for _, promo in promos_df.iterrows():
        start_date = pd.to_datetime(promo['Start'])
        end_date = pd.to_datetime(promo['End'])
        applies_to = str(promo.get('Applies To', '')).lower()
        
        raw_discount = promo.get('Discount', 0.0)
        
        # Safely parse discount percentage string
        if isinstance(raw_discount, str):
            if '%' in raw_discount:
                discount_rate = float(raw_discount.replace('%', '')) / 100.0
            else:
                try:
                    discount_rate = float(raw_discount)
                except ValueError:
                    discount_rate = 0.0
        else:
            discount_rate = float(raw_discount)
            if discount_rate > 1.0:
                discount_rate = discount_rate / 100.0

        # Create date mask
        mask = (combined_df['order_date'] >= start_date) & (combined_df['order_date'] <= end_date)

        # Refine by product class applicability
        if 'mountain' in applies_to and 'class' in combined_df.columns:
            mask &= (combined_df['class'].str.strip().str.upper() == 'MTB')
        elif 'road' in applies_to and 'class' in combined_df.columns:
            mask &= (combined_df['class'].str.strip().str.upper() == 'ROAD')

        # Apply promotional flags and discounts in bulk
        combined_df.loc[mask, 'is_promo_active'] = True
        combined_df.loc[mask, 'promo_discount'] = discount_rate

    return combined_df


def standardize_and_merge_channels(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Standardizes schema definitions across DTC, dealer shipments, and POS logs,
    dynamically mapping units and revenue, and applies promotion matching.
    """
    standardized_dfs = []

    for channel_key in ['dtc', 'dealers', 'pos']:
        df = raw_data.get(channel_key)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Standardize date column
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col:
            df['order_date'] = pd.to_datetime(df[date_col])
        else:
            raise KeyError(f"Could not identify a date column in channel {channel_key}")

        # Standardize channel name labeling to match notebook expectations
        if 'channel' not in df.columns:
            if channel_key == 'dtc':
                df['channel'] = 'DTC Web'
            elif channel_key == 'dealers':
                df['channel'] = 'Dealer Wholesale'
            elif channel_key == 'pos':
                df['channel'] = 'Commercial PO'
            else:
                df['channel'] = channel_key.upper()

        # Dynamically locate or normalize units sold / quantity
        units_col = next((col for col in df.columns if col.lower() in ['units_sold', 'quantity', 'qty', 'units']), None)
        if units_col and units_col != 'units_sold':
            df['units_sold'] = df[units_col]
        elif not units_col:
            df['units_sold'] = 1

        # Dynamically locate or compute gross revenue / total amount
        rev_col = next((col for col in df.columns if any(term in col.lower() for term in ['gross_revenue', 'revenue', 'total_amount', 'total', 'amount'])), None)
        if rev_col and rev_col != 'gross_revenue':
            df['gross_revenue'] = df[rev_col]
        elif not rev_col:
            # Fallback: check if unit price and quantity exist to compute revenue
            price_col = next((col for col in df.columns if any(term in col.lower() for term in ['price', 'unit_price', 'msrp'])), None)
            if price_col:
                df['gross_revenue'] = df['units_sold'] * df[price_col]
            else:
                df['gross_revenue'] = 0.0

        standardized_dfs.append(df)

    if not standardized_dfs:
        raise ValueError("No transaction datasets were successfully loaded or standardized.")

    combined_df = pd.concat(standardized_dfs, ignore_index=True)

    # Apply vectorized promotion logic
    promos_df = raw_data.get('promos', pd.DataFrame()).copy()
    if not promos_df.empty:
        logging.info("Mapping promotional periods to combined transactions...")
        combined_df = apply_promotions(combined_df, promos_df)
    else:
        combined_df['is_promo_active'] = False
        combined_df['promo_discount'] = 0.0

    return combined_df

    return combined_df


def aggregate_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction logs into weekly buckets across product and region dimensions,
    matching the notebook schema.
    """
    df['week_start_date'] = df['order_date'].dt.to_period('W').dt.start_time

    # Identify dimension columns present in the dataset
    possible_dims = ['channel', 'region', 'sku', 'class', 'size', 'trim']
    group_dims = ['week_start_date'] + [col for col in possible_dims if col in df.columns]

    agg_dict = {
        'units_sold': 'sum',
        'gross_revenue': 'sum',
        'is_promo_active': 'max',
        'promo_discount': 'max'
    }

    weekly_df = (
        df.groupby(group_dims, as_index=False)
        .agg(agg_dict)
    )

    # Calculate average selling price correctly
    weekly_df['avg_selling_price'] = np.where(
        weekly_df['units_sold'] > 0,
        weekly_df['gross_revenue'] / weekly_df['units_sold'],
        0.0
    )

    # Reorder columns to match the notebook screenshot structure precisely
    preferred_order = [
        'week_start_date', 'channel', 'region', 'sku', 'class', 'size', 'trim',
        'units_sold', 'gross_revenue', 'avg_selling_price', 'is_promo_active', 'promo_discount'
    ]
    existing_cols = [col for col in preferred_order if col in weekly_df.columns]
    remaining_cols = [col for col in weekly_df.columns if col not in existing_cols]
    
    weekly_df = weekly_df[existing_cols + remaining_cols].sort_values(by=group_dims).reset_index(drop=True)

    return weekly_df


def run_etl_pipeline() -> pd.DataFrame:
    """
    Executes the complete ingestion, standardization, and aggregation pipeline,
    saving the result to the designated processed CSV path.
    """
    logging.info("Starting ETL pipeline...")
    raw_data = load_raw_data()
    standardized_df = standardize_and_merge_channels(raw_data)
    weekly_sales_df = aggregate_weekly_sales(standardized_df)

    logging.info(f"Saving processed master sales file to: {PROCESSED_SALES_FILE}")
    weekly_sales_df.to_csv(PROCESSED_SALES_FILE, index=False)
    logging.info("ETL pipeline completed successfully.")

    return weekly_sales_df


if __name__ == "__main__":
    run_etl_pipeline()