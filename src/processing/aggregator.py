"""Aggregation for creating daily customer metrics."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataAggregator:
    
    def __init__(self, config: dict):
        self.config = config
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Aggregating {len(df):,} transactions")
        
        # Ensure we have the required columns
        required_cols = ['file_date', 'customer_id', 'unit_price_gbp', 'quantity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create revenue column: quantity * unit_price_gbp
        df['revenue_gbp'] = df['quantity'] * df['unit_price_gbp']
        
        # Separate positive (purchases) and negative (returns) quantities
        df['is_return'] = df['quantity'] < 0
        df['abs_quantity'] = df['quantity'].abs()
        
        # Group by date and customer
        grouped = df.groupby(['file_date', 'customer_id'])
        
        # Aggregate metrics
        aggregated = grouped.agg({
            'invoice_id': 'nunique',  # Distinct invoices
            'abs_quantity': 'sum',  # Total items (absolute value)
            'product_id': 'nunique',  # Unique products
            'product_category': 'nunique' if 'product_category' in df.columns else 'count',  # Unique categories
        }).reset_index()
        
        # Rename columns
        aggregated.columns = ['date', 'customer_id', 'orders', 'items', 'unique_products', 'unique_categories']
        
        # Calculate gross revenue (positive quantities only)
        gross = df[df['quantity'] > 0].groupby(['file_date', 'customer_id'])['revenue_gbp'].sum().reset_index()
        gross.columns = ['date', 'customer_id', 'gross_gbp']
        
        # Calculate returns (negative quantities)
        returns = df[df['quantity'] < 0].groupby(['file_date', 'customer_id'])['revenue_gbp'].sum().reset_index()
        returns.columns = ['date', 'customer_id', 'returns_gbp']
        
        # Merge all metrics
        aggregated = aggregated.merge(gross, on=['date', 'customer_id'], how='left')
        aggregated = aggregated.merge(returns, on=['date', 'customer_id'], how='left')
        
        # Fill NaN values (customers with no returns have returns_gbp = 0, etc.)
        aggregated['gross_gbp'] = aggregated['gross_gbp'].fillna(0)
        aggregated['returns_gbp'] = aggregated['returns_gbp'].fillna(0)
        
        # Calculate net revenue
        aggregated['net_gbp'] = aggregated['gross_gbp'] + aggregated['returns_gbp']
        
        # Calculate average order value
        aggregated['avg_order_value'] = aggregated['net_gbp'] / aggregated['orders']
        aggregated['avg_order_value'] = aggregated['avg_order_value'].replace([np.inf, -np.inf], 0)
        
        # Ensure date is in proper format
        aggregated['date'] = pd.to_datetime(aggregated['date'])
        
        # Sort by date and customer
        aggregated = aggregated.sort_values(['date', 'customer_id']).reset_index(drop=True)
        
        logger.info(f"Aggregation complete: {len(aggregated):,} daily customer records")
        self._log_aggregation_stats(aggregated)
        
        return aggregated
    
    def _log_aggregation_stats(self, df: pd.DataFrame):
        logger.info(f"\nAggregated {len(df):,} records for {df['customer_id'].nunique():,} customers")
        logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"Total net revenue: £{df['net_gbp'].sum():,.2f} (£{df['net_gbp'].mean():,.2f}/day avg)")
    
    def save(self, df: pd.DataFrame, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved aggregated metrics to {output_path} (Parquet format)")
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
            logger.info(f"Saved aggregated metrics to {output_path} (CSV format)")
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")


def aggregate_transactions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    aggregator = DataAggregator(config)
    return aggregator.aggregate(df)


def load_aggregated_metrics(file_path: str) -> pd.DataFrame:
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path, parse_dates=['date'])
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(df):,} records from {file_path}")
    
    return df




