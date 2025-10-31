"""FX conversion utilities for normalizing to GBP."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def merge_fx_rates(
    transactions_df: pd.DataFrame, 
    fx_rates_df: pd.DataFrame,
    date_column: str = 'file_date'
) -> pd.DataFrame:
    # Ensure date columns are strings for merging
    transactions_df = transactions_df.copy()
    fx_rates_df = fx_rates_df.copy()
    
    # Ensure date format consistency
    transactions_df[date_column] = pd.to_datetime(transactions_df[date_column]).dt.strftime('%Y-%m-%d')
    fx_rates_df['date'] = pd.to_datetime(fx_rates_df['date']).dt.strftime('%Y-%m-%d')
    
    # Merge on date and currency
    merged_df = transactions_df.merge(
        fx_rates_df[['date', 'currency', 'rate_to_gbp']],
        left_on=[date_column, 'currency'],
        right_on=['date', 'currency'],
        how='left'
    )
    
    # Check for missing FX rates
    missing_rates = merged_df['rate_to_gbp'].isnull().sum()
    if missing_rates > 0:
        logger.warning(f"Missing FX rates for {missing_rates} transactions ({missing_rates/len(merged_df)*100:.2f}%)")
        
        # Try to fill with nearest available rate for same currency
        merged_df = _fill_missing_fx_rates(merged_df, fx_rates_df, date_column)
    
    logger.info(f"Merged FX rates: {len(merged_df):,} transactions")
    
    return merged_df


def _fill_missing_fx_rates(
    transactions_df: pd.DataFrame,
    fx_rates_df: pd.DataFrame,
    date_column: str
) -> pd.DataFrame:
    # Fill missing rates with nearest available rate for same currency
    transactions_df = transactions_df.copy()
    
    # Convert dates to datetime for comparison
    fx_rates_df['date_dt'] = pd.to_datetime(fx_rates_df['date'])
    
    # For each row with missing rate
    missing_mask = transactions_df['rate_to_gbp'].isnull()
    
    for idx in transactions_df[missing_mask].index:
        currency = transactions_df.loc[idx, 'currency']
        date_str = transactions_df.loc[idx, date_column]
        target_date = pd.to_datetime(date_str)
        
        # Get all rates for this currency
        currency_rates = fx_rates_df[fx_rates_df['currency'] == currency].copy()
        
        if len(currency_rates) > 0:
            # Find nearest date
            currency_rates['date_diff'] = abs((currency_rates['date_dt'] - target_date).dt.days)
            nearest = currency_rates.loc[currency_rates['date_diff'].idxmin()]
            
            transactions_df.loc[idx, 'rate_to_gbp'] = nearest['rate_to_gbp']
            
            if nearest['date_diff'] > 0:
                logger.debug(f"Filled rate for {date_str} {currency} using rate from {nearest['date']} ({nearest['date_diff']} days away)")
    
    # Log remaining missing rates
    still_missing = transactions_df['rate_to_gbp'].isnull().sum()
    if still_missing > 0:
        logger.warning(f"Could not fill {still_missing} FX rates - will drop these rows")
    
    return transactions_df


def convert_to_gbp(
    transactions_df: pd.DataFrame,
    price_column: str = 'unit_price',
    output_column: str = 'unit_price_gbp'
) -> pd.DataFrame:
    transactions_df = transactions_df.copy()
    
    # Convert to GBP
    transactions_df[output_column] = transactions_df[price_column] * transactions_df['rate_to_gbp']
    
    # Log conversion stats
    converted = transactions_df[output_column].notna().sum()
    logger.info(f"Converted {converted:,} prices to GBP")
    
    if price_column in transactions_df.columns:
        original_total = (transactions_df[price_column] * transactions_df.get('quantity', 1)).sum()
        gbp_total = (transactions_df[output_column] * transactions_df.get('quantity', 1)).sum()
        logger.info(f"Total value: {original_total:,.2f} (mixed currencies) -> {gbp_total:,.2f} GBP")
    
    return transactions_df


def apply_fx_conversion(
    transactions_df: pd.DataFrame,
    fx_rates_df: pd.DataFrame,
    date_column: str = 'file_date',
    price_column: str = 'unit_price',
    output_column: str = 'unit_price_gbp'
) -> pd.DataFrame:
    df = merge_fx_rates(transactions_df, fx_rates_df, date_column)
    df = convert_to_gbp(df, price_column, output_column)
    
    # Drop rows without FX rates
    initial_rows = len(df)
    df = df.dropna(subset=['rate_to_gbp'])
    dropped = initial_rows - len(df)
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows ({dropped/initial_rows*100:.2f}%) due to missing FX rates")
    
    return df


def get_conversion_summary(transactions_df: pd.DataFrame) -> dict:
    summary = {
        'total_rows': len(transactions_df),
        'rows_with_gbp': transactions_df['unit_price_gbp'].notna().sum(),
        'currencies': transactions_df['currency'].value_counts().to_dict(),
        'avg_rate_to_gbp': transactions_df.groupby('currency')['rate_to_gbp'].mean().to_dict(),
    }
    
    return summary




