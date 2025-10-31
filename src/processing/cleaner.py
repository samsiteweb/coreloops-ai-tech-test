"""Data cleaning for duplicates and missing values."""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)


class DataCleaner:
    
    def __init__(self, config: dict):
        self.config = config
        self.dedup_columns = config['processing']['deduplication_columns']
        self.imputation_window = config['processing']['imputation_window_days']
        self.stats = {
            'initial_rows': 0,
            'duplicates_removed': 0,
            'missing_customer_id_dropped': 0,
            'missing_price_imputed': 0,
            'missing_price_dropped': 0,
            'final_rows': 0
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        self.stats['initial_rows'] = len(df)
        logger.info(f"Cleaning {len(df):,} rows...")
        
        df = self.remove_duplicates(df)
        df = self.handle_missing_customer_id(df)
        df = self.handle_missing_price(df)
        df = self.handle_missing_description(df)
        df = self.validate_data(df)
        
        self.stats['final_rows'] = len(df)
        self.log_summary()
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        # Get columns that exist in the dataframe
        available_columns = [col for col in self.dedup_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No deduplication columns found in data")
            return df
        
        # Identify duplicates
        duplicates_mask = df.duplicated(subset=available_columns, keep='first')
        duplicates_count = duplicates_mask.sum()
        
        # Remove duplicates
        df = df[~duplicates_mask].copy()
        
        self.stats['duplicates_removed'] = duplicates_count
        
        if duplicates_count > 0:
            logger.info(f"Removed {duplicates_count:,} duplicate rows ({duplicates_count/initial_count*100:.2f}%)")
        else:
            logger.info("No duplicates found")
        
        return df
    
    def handle_missing_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows without customer_id since we need it for aggregation
        if 'customer_id' not in df.columns:
            logger.warning("customer_id column not found")
            return df
        
        missing_count = df['customer_id'].isnull().sum()
        
        if missing_count > 0:
            df = df[df['customer_id'].notna()].copy()
            self.stats['missing_customer_id_dropped'] = missing_count
            logger.info(f"Dropped {missing_count:,} rows with missing customer_id ({missing_count/(len(df)+missing_count)*100:.2f}%)")
        else:
            logger.info("No missing customer_id values")
        
        return df
    
    def handle_missing_price(self, df: pd.DataFrame) -> pd.DataFrame:
        # Impute with median price for same product/currency, drop if no history
        if 'unit_price' not in df.columns:
            logger.warning("unit_price column not found")
            return df
        
        missing_mask = df['unit_price'].isnull()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            logger.info("No missing unit_price values")
            return df
        
        logger.info(f"Found {missing_count:,} missing unit_price values, attempting imputation...")
        
        # Ensure we have file_date for temporal imputation
        if 'file_date' in df.columns:
            df['file_date_dt'] = pd.to_datetime(df['file_date'])
        
        imputed_count = 0
        
        # For each missing price, try to impute
        for idx in df[missing_mask].index:
            product_id = df.loc[idx, 'product_id'] if 'product_id' in df.columns else None
            currency = df.loc[idx, 'currency'] if 'currency' in df.columns else None
            
            if product_id is None or currency is None:
                continue
            
            # Get historical prices for this product and currency
            historical_mask = (
                (df['product_id'] == product_id) & 
                (df['currency'] == currency) & 
                (df['unit_price'].notna())
            )
            
            # If we have file_date, limit to nearby dates
            if 'file_date_dt' in df.columns:
                target_date = df.loc[idx, 'file_date_dt']
                date_diff = abs((df['file_date_dt'] - target_date).dt.days)
                historical_mask &= (date_diff <= self.imputation_window)
            
            historical_prices = df.loc[historical_mask, 'unit_price']
            
            if len(historical_prices) > 0:
                median_price = historical_prices.median()
                df.loc[idx, 'unit_price'] = median_price
                imputed_count += 1
        
        # Drop remaining rows with missing prices
        still_missing = df['unit_price'].isnull().sum()
        df = df[df['unit_price'].notna()].copy()
        
        self.stats['missing_price_imputed'] = imputed_count
        self.stats['missing_price_dropped'] = still_missing
        
        logger.info(f"Imputed {imputed_count:,} prices using product median")
        if still_missing > 0:
            logger.info(f"Dropped {still_missing:,} rows with no historical price data")
        
        return df
    
    def handle_missing_description(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep rows with missing description - not critical for analysis
        if 'description' not in df.columns:
            return df
        
        missing_count = df['description'].isnull().sum()
        
        if missing_count > 0:
            logger.info(f"Keeping {missing_count:,} rows with missing description ({missing_count/len(df)*100:.2f}%)")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validating data quality...")
        
        # Check for negative prices (should not exist)
        if 'unit_price' in df.columns:
            negative_prices = (df['unit_price'] < 0).sum()
            if negative_prices > 0:
                logger.warning(f"Found {negative_prices} rows with negative unit_price (unusual)")
        
        # Check for zero prices (might be valid for free items)
        if 'unit_price' in df.columns:
            zero_prices = (df['unit_price'] == 0).sum()
            if zero_prices > 0:
                logger.info(f"Found {zero_prices} rows with zero unit_price (may be promotions)")
        
        # Check for extremely high prices (potential data errors)
        if 'unit_price' in df.columns:
            price_99th = df['unit_price'].quantile(0.99)
            high_prices = (df['unit_price'] > price_99th * 10).sum()
            if high_prices > 0:
                logger.warning(f"Found {high_prices} rows with extremely high prices (>10x 99th percentile)")
        
        # Check quantity range
        if 'quantity' in df.columns:
            logger.info(f"Quantity range: {df['quantity'].min()} to {df['quantity'].max()}")
            returns_count = (df['quantity'] < 0).sum()
            logger.info(f"Returns (negative quantity): {returns_count:,} rows ({returns_count/len(df)*100:.2f}%)")
        
        return df
    
    def log_summary(self):
        logger.info(f"\nCleaning summary: {self.stats['initial_rows']:,} -> {self.stats['final_rows']:,} rows ({self.stats['final_rows']/self.stats['initial_rows']*100:.2f}% retained)")
        logger.info(f"  Removed: {self.stats['duplicates_removed']:,} duplicates, {self.stats['missing_customer_id_dropped']:,} missing customer_id")
        logger.info(f"  Imputed: {self.stats['missing_price_imputed']:,} prices, dropped: {self.stats['missing_price_dropped']:,}")
    
    def get_stats(self) -> Dict:
        return self.stats.copy()


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cleaner = DataCleaner(config)
    return cleaner.clean(df)




