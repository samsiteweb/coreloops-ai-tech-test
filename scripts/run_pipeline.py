"""ETL pipeline for processing transaction data."""

import sys
import logging
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_all_data
from src.ingestion.fx_converter import apply_fx_conversion
from src.processing.cleaner import clean_data
from src.processing.aggregator import DataAggregator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    logger.info("Starting ETL pipeline...")
    
    config = load_config()
    
    logger.info("Loading data from GCS...")
    transactions_df, fx_rates_df = load_all_data(config)
    logger.info(f"Loaded {len(transactions_df):,} transactions and {len(fx_rates_df):,} FX rates")
    
    logger.info("\nConverting to GBP...")
    transactions_df = apply_fx_conversion(
        transactions_df,
        fx_rates_df,
        date_column='file_date',
        price_column='unit_price',
        output_column='unit_price_gbp'
    )
    
    logger.info("\nCleaning data...")
    transactions_df = clean_data(transactions_df, config)
    
    logger.info("\nAggregating daily metrics...")
    aggregator = DataAggregator(config)
    metrics_df = aggregator.aggregate(transactions_df)
    
    logger.info("\nSaving...")
    output_path = config['output']['metrics_file']
    aggregator.save(metrics_df, output_path)
    
    logger.info(f"\nDone. Saved to: {output_path}")
    logger.info(f"Next: python scripts/train_model.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)




