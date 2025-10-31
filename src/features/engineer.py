"""Feature engineering for time-series prediction."""

import logging
import pandas as pd
import numpy as np
from datetime import timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    
    def __init__(self, config: dict):
        self.config = config
        self.rolling_windows = config['features']['rolling_windows']
        self.lag_days = config['features']['lag_days']
        self.min_history_days = config['features']['min_history_days']
    
    def create_features(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building features from {len(metrics_df):,} records")
        
        # Ensure date is datetime
        metrics_df = metrics_df.copy()
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        
        # Sort by customer and date
        metrics_df = metrics_df.sort_values(['customer_id', 'date']).reset_index(drop=True)
        
        # Create features
        df = metrics_df.copy()
        
        # Rolling window features
        df = self._add_rolling_features(df)
        
        # Lag features
        df = self._add_lag_features(df)
        
        # Recency features
        df = self._add_recency_features(df)
        
        # Frequency features
        df = self._add_frequency_features(df)
        
        # Trend features
        df = self._add_trend_features(df)
        
        # Temporal features
        df = self._add_temporal_features(df)
        
        # Customer behavior features
        df = self._add_behavior_features(df)
        
        # Create target: next day's net_gbp
        df = self._create_target(df)
        
        # Drop rows with insufficient history
        df = self._filter_insufficient_history(df)
        
        logger.info(f"Done: {len(df):,} records with {len(df.columns)} features")
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding rolling features...")
        
        for window in self.rolling_windows:
            # Rolling mean
            df[f'net_gbp_rolling_mean_{window}d'] = df.groupby('customer_id')['net_gbp'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            df[f'orders_rolling_mean_{window}d'] = df.groupby('customer_id')['orders'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            df[f'items_rolling_mean_{window}d'] = df.groupby('customer_id')['items'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling sum
            df[f'net_gbp_rolling_sum_{window}d'] = df.groupby('customer_id')['net_gbp'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
            )
            
            # Rolling std (volatility)
            df[f'net_gbp_rolling_std_{window}d'] = df.groupby('customer_id')['net_gbp'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
            )
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding lag features...")
        
        for lag in self.lag_days:
            df[f'net_gbp_lag_{lag}d'] = df.groupby('customer_id')['net_gbp'].shift(lag)
            df[f'orders_lag_{lag}d'] = df.groupby('customer_id')['orders'].shift(lag)
        
        return df
    
    def _add_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding recency features...")
        
        # Days since last transaction (for each customer)
        df['prev_date'] = df.groupby('customer_id')['date'].shift(1)
        df['days_since_last_transaction'] = (df['date'] - df['prev_date']).dt.days
        
        # Fill first transaction for each customer
        df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(999)
        
        # Days since first transaction (customer age in days)
        df['first_transaction_date'] = df.groupby('customer_id')['date'].transform('first')
        df['customer_age_days'] = (df['date'] - df['first_transaction_date']).dt.days
        
        df = df.drop(columns=['prev_date', 'first_transaction_date'])
        
        return df
    
    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding frequency features...")
        
        # Transaction count in past windows
        for window in self.rolling_windows:
            df[f'transaction_count_{window}d'] = df.groupby('customer_id')['net_gbp'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).count()
            )
        
        # Cumulative transaction count
        df['cumulative_transactions'] = df.groupby('customer_id').cumcount()
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding trend features...")
        
        # Week-over-week change
        df['net_gbp_7d_avg'] = df.groupby('customer_id')['net_gbp'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
        df['net_gbp_14d_avg'] = df.groupby('customer_id')['net_gbp'].transform(
            lambda x: x.shift(8).rolling(window=7, min_periods=1).mean()
        )
        
        # Growth rate (avoid division by zero)
        df['net_gbp_growth_rate'] = (df['net_gbp_7d_avg'] - df['net_gbp_14d_avg']) / (df['net_gbp_14d_avg'] + 1)
        df['net_gbp_growth_rate'] = df['net_gbp_growth_rate'].replace([np.inf, -np.inf], 0)
        
        df = df.drop(columns=['net_gbp_7d_avg', 'net_gbp_14d_avg'])
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding temporal features...")
        
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
        
        return df
    
    def _add_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding behavior features...")
        
        # Average items per order (avoid division by zero)
        df['avg_items_per_order'] = df['items'] / (df['orders'] + 0.001)
        
        # Return rate (returns / gross)
        df['return_rate'] = -df['returns_gbp'] / (df['gross_gbp'] + 1)
        df['return_rate'] = df['return_rate'].fillna(0).clip(0, 1)
        
        # Product diversity (rolling)
        df['unique_products_rolling_30d'] = df.groupby('customer_id')['unique_products'].transform(
            lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
        )
        
        return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Shift net_gbp backward (so we predict the current row's value)
        df['target_net_gbp'] = df.groupby('customer_id')['net_gbp'].shift(-1)
        
        # Mark rows where we have a target
        df['has_target'] = df['target_net_gbp'].notna()
        
        return df
    
    def _filter_insufficient_history(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        # Keep only rows where customer has at least min_history_days of data
        df = df[df['customer_age_days'] >= self.min_history_days].copy()
        
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count:,} rows with insufficient history (<{self.min_history_days} days)")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        # Columns to exclude
        exclude_cols = [
            'date', 'customer_id', 'target_net_gbp', 'has_target',
            'net_gbp', 'gross_gbp', 'returns_gbp',  # Raw metrics (not features)
            'orders', 'items', 'unique_products', 'unique_categories',  # Raw metrics
            'avg_order_value'  # Derived but not a feature
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Identified {len(feature_cols)} feature columns")
        
        return feature_cols


def engineer_features(metrics_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    engineer = FeatureEngineer(config)
    return engineer.create_features(metrics_df)


def get_feature_names(config: dict) -> list:
    rolling_windows = config['features']['rolling_windows']
    lag_days = config['features']['lag_days']
    
    features = []
    
    # Rolling features
    for window in rolling_windows:
        features.extend([
            f'net_gbp_rolling_mean_{window}d',
            f'net_gbp_rolling_sum_{window}d',
            f'net_gbp_rolling_std_{window}d',
            f'orders_rolling_mean_{window}d',
            f'items_rolling_mean_{window}d',
        ])
    
    # Lag features
    for lag in lag_days:
        features.extend([
            f'net_gbp_lag_{lag}d',
            f'orders_lag_{lag}d',
        ])
    
    # Other features
    features.extend([
        'days_since_last_transaction',
        'customer_age_days',
        'cumulative_transactions',
        'net_gbp_growth_rate',
        'day_of_week',
        'day_of_month',
        'is_weekend',
        'is_month_start',
        'is_month_end',
        'avg_items_per_order',
        'return_rate',
        'unique_products_rolling_30d',
    ])
    
    # Add transaction count features
    for window in rolling_windows:
        features.append(f'transaction_count_{window}d')
    
    return features




