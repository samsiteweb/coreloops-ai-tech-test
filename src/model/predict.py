"""Prediction module for customer spending forecasts."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .train import ModelLoader

logger = logging.getLogger(__name__)


class Predictor:
    
    def __init__(self, model_dir: str, metrics_df: pd.DataFrame):
        self.model_loader = ModelLoader(model_dir)
        self.metrics_df = metrics_df.copy()
        self.metrics_df['date'] = pd.to_datetime(self.metrics_df['date'])
        
        logger.info(f"Predictor initialized with {len(metrics_df):,} historical records")
    
    def predict_for_customer_date(
        self, 
        customer_id: str, 
        target_date: str,
        feature_engineer
    ) -> dict:
        target_date = pd.to_datetime(target_date)
        
        logger.info(f"Predicting for customer {customer_id} on {target_date.date()}")
        
        # Get historical data for this customer up to (but not including) target date
        customer_history = self.metrics_df[
            (self.metrics_df['customer_id'] == customer_id) &
            (self.metrics_df['date'] < target_date)
        ].copy()
        
        if len(customer_history) == 0:
            logger.warning(f"No historical data found for customer {customer_id}")
            return {
                'customer_id': customer_id,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'prediction': 0.0,
                'error': 'No historical data available',
                'confidence': 'low'
            }
        
        # Create a dummy row for the target date (we'll generate features for it)
        # Use the last known values as placeholders
        last_record = customer_history.iloc[-1]
        dummy_row = last_record.copy()
        dummy_row['date'] = target_date
        
        # Append dummy row to history
        full_history = pd.concat([customer_history, pd.DataFrame([dummy_row])], ignore_index=True)
        
        # Generate features
        features_df = feature_engineer.create_features(full_history)
        
        # Get the row for the target date
        target_row = features_df[features_df['date'] == target_date]
        
        if len(target_row) == 0:
            logger.warning(f"Could not generate features for {customer_id} on {target_date.date()}")
            return {
                'customer_id': customer_id,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'prediction': 0.0,
                'error': 'Insufficient history for feature generation',
                'confidence': 'low'
            }
        
        # Prepare features
        X = target_row[self.model_loader.feature_columns]
        
        # Make prediction
        prediction = self.model_loader.predict(X)[0]
        
        # Calculate confidence based on history
        history_days = (target_date - customer_history['date'].min()).days
        recent_transactions = len(customer_history[customer_history['date'] > (target_date - timedelta(days=30))])
        
        if history_days >= 30 and recent_transactions >= 5:
            confidence = 'high'
        elif history_days >= 14 and recent_transactions >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        result = {
            'customer_id': customer_id,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'prediction': float(max(0, prediction)),  # Ensure non-negative
            'confidence': confidence,
            'history_days': int(history_days),
            'recent_transactions_30d': int(recent_transactions)
        }
        
        logger.info(f"Prediction: £{result['prediction']:.2f} (confidence: {confidence})")
        
        return result
    
    def predict_batch(
        self,
        predictions_list: list,
        feature_engineer
    ) -> pd.DataFrame:
        results = []
        
        for item in predictions_list:
            result = self.predict_for_customer_date(
                customer_id=item['customer_id'],
                target_date=item['target_date'],
                feature_engineer=feature_engineer
            )
            results.append(result)
        
        return pd.DataFrame(results)


def make_prediction(
    customer_id: str,
    target_date: str,
    model_dir: str,
    metrics_df: pd.DataFrame,
    feature_engineer
) -> dict:
    predictor = Predictor(model_dir, metrics_df)
    return predictor.predict_for_customer_date(customer_id, target_date, feature_engineer)




