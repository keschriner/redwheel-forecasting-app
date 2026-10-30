import argparse
import logging
from src.config import configure_pandas
from src.etl import run_etl_pipeline
from src.forecasting import run_forecasting_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Apply global configurations
    configure_pandas()
    
    parser = argparse.ArgumentParser(description="Bicycle Sales Ingestion and Forecasting Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline command to execute")
    
    # Subcommand: ingest
    subparsers.add_parser("ingest", help="Read raw Atomic CSVs, standardize schemas, apply promotions, and save master weekly sales.")
    
    # Subcommand: forecast
    subparsers.add_parser("forecast", help="Run statsmodels seasonal decomposition and Prophet time-series forecasting models.")
    
    # Subcommand: all (runs both sequentially)
    subparsers.add_parser("all", help="Run both the ingestion and forecasting pipelines sequentially.")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        logging.info("Starting ingestion step...")
        run_etl_pipeline()
    elif args.command == "forecast":
        logging.info("Starting forecasting step...")
        run_forecasting_pipeline()
    elif args.command == "all":
        logging.info("Starting full end-to-end pipeline...")
        run_etl_pipeline()
        run_forecasting_pipeline()

if __name__ == "__main__":
    main()