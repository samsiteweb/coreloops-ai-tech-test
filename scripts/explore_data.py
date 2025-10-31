"""Data exploration and profiling script."""

import pandas as pd
import requests
import yaml
import os
import logging
from pathlib import Path

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


def download_file(url, local_path):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded: {url} -> {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def explore_data():
    config = load_config()
    base_url = config['data']['gcs_base_url']
    local_data_dir = config['data']['local_data_dir']
    
    # Create data directory
    os.makedirs(local_data_dir, exist_ok=True)
    
    # Download FX rates
    fx_url = f"{base_url}/{config['data']['fx_rates_file']}"
    fx_local_path = f"{local_data_dir}/{config['data']['fx_rates_file']}"
    download_file(fx_url, fx_local_path)
    
    # Download sample transaction files (first few days)
    sample_dates = ['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05']
    
    all_transactions = []
    for date in sample_dates:
        url = f"{base_url}/data/{date}.csv"
        local_path = f"{local_data_dir}/{date}.csv"
        if download_file(url, local_path):
            df = pd.read_csv(local_path)
            df['file_date'] = date
            all_transactions.append(df)
    
    if not all_transactions:
        logger.error("No data downloaded. Check URLs and network connection.")
        return
    
    # Combine all transactions
    df = pd.concat(all_transactions, ignore_index=True)
    
    logger.info("\n--- Data Exploration ---")
    
    # Basic info
    logger.info(f"\nTotal rows: {len(df):,}")
    logger.info(f"Date range: {df['file_date'].min()} to {df['file_date'].max()}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Missing values
    logger.info("\n--- MISSING VALUES ---")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    for col, pct in missing_pct[missing_pct > 0].items():
        logger.info(f"{col}: {pct}% ({df[col].isnull().sum():,} rows)")
    
    # Duplicates
    logger.info("\n--- DUPLICATES ---")
    dup_cols = ['invoice_id', 'product_id', 'quantity', 'unit_price', 'customer_id', 'timestamp']
    available_dup_cols = [col for col in dup_cols if col in df.columns]
    duplicates = df.duplicated(subset=available_dup_cols, keep=False)
    logger.info(f"Duplicate rows: {duplicates.sum():,} ({duplicates.sum()/len(df)*100:.2f}%)")
    
    # Currency distribution
    if 'currency' in df.columns:
        logger.info("\n--- CURRENCY DISTRIBUTION ---")
        logger.info(df['currency'].value_counts().to_string())
    
    # Quantity statistics (returns vs purchases)
    if 'quantity' in df.columns:
        logger.info("\n--- QUANTITY STATISTICS ---")
        positive_qty = df[df['quantity'] > 0]
        negative_qty = df[df['quantity'] < 0]
        logger.info(f"Positive quantities (purchases): {len(positive_qty):,} ({len(positive_qty)/len(df)*100:.2f}%)")
        logger.info(f"Negative quantities (returns): {len(negative_qty):,} ({len(negative_qty)/len(df)*100:.2f}%)")
        logger.info(f"Quantity range: {df['quantity'].min()} to {df['quantity'].max()}")
    
    # Price statistics
    if 'unit_price' in df.columns:
        logger.info("\n--- PRICE STATISTICS ---")
        logger.info(f"Price range: {df['unit_price'].min():.2f} to {df['unit_price'].max():.2f}")
        logger.info(f"Mean price: {df['unit_price'].mean():.2f}")
        logger.info(f"Median price: {df['unit_price'].median():.2f}")
    
    # Customer statistics
    if 'customer_id' in df.columns:
        non_null_customers = df[df['customer_id'].notna()]
        logger.info("\n--- CUSTOMER STATISTICS ---")
        logger.info(f"Unique customers: {non_null_customers['customer_id'].nunique():,}")
        logger.info(f"Transactions per customer (avg): {len(non_null_customers) / non_null_customers['customer_id'].nunique():.2f}")
    
    # Product statistics
    if 'product_id' in df.columns:
        logger.info("\n--- PRODUCT STATISTICS ---")
        logger.info(f"Unique products: {df['product_id'].nunique():,}")
        if 'product_category' in df.columns:
            logger.info(f"Unique categories: {df['product_category'].nunique()}")
            logger.info("\nTop 5 categories:")
            logger.info(df['product_category'].value_counts().head().to_string())
    
    # FX Rates
    logger.info("\n--- FX RATES ---")
    if os.path.exists(fx_local_path):
        fx_df = pd.read_csv(fx_local_path)
        logger.info(f"FX rates shape: {fx_df.shape}")
        logger.info(f"Currencies: {fx_df['currency'].unique() if 'currency' in fx_df.columns else 'N/A'}")
        logger.info(f"Date range: {fx_df['date'].min() if 'date' in fx_df.columns else 'N/A'} to {fx_df['date'].max() if 'date' in fx_df.columns else 'N/A'}")
        logger.info("\nSample FX rates:")
        logger.info(fx_df.head(10).to_string())
    
    return df


if __name__ == "__main__":
    explore_data()




