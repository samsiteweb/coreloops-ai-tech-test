"""Model training with time-based validation."""

import logging
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.model = None
        self.feature_columns = None
        self.metrics = {}
    
    def train(self, features_df: pd.DataFrame, feature_columns: list) -> dict:
        logger.info("Training...")
        
        self.feature_columns = feature_columns
        
        # Filter to rows with targets
        df = features_df[features_df['has_target']].copy()
        logger.info(f"Training data: {len(df):,} samples with targets")
        
        # Time-based split
        train_df, val_df, test_df = self._time_based_split(df)
        
        # Prepare data
        X_train, y_train = self._prepare_data(train_df, feature_columns)
        X_val, y_val = self._prepare_data(val_df, feature_columns)
        X_test, y_test = self._prepare_data(test_df, feature_columns)
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Build and train model
        self.model = self._build_model()
        
        logger.info(f"Training {self.model_config['type']}...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on all sets
        metrics = {}
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        metrics['train'] = self._calculate_metrics(y_train, train_pred, 'Training')
        
        # Validation metrics
        val_pred = self.model.predict(X_val)
        metrics['validation'] = self._calculate_metrics(y_val, val_pred, 'Validation')
        
        # Test metrics
        test_pred = self.model.predict(X_test)
        metrics['test'] = self._calculate_metrics(y_test, test_pred, 'Test')
        
        self.metrics = metrics
        
        # Feature importance (if available)
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            self._log_feature_importance()
        
        logger.info("Model training complete")
        
        return metrics
    
    def _time_based_split(self, df: pd.DataFrame) -> tuple:
        df = df.sort_values('date').reset_index(drop=True)
        
        # Get split sizes from config
        test_size = self.model_config['test_size']
        val_size = self.model_config['validation_size']
        
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size / (1 - test_size)))
        
        train_df = df.iloc[:val_idx].copy()
        val_df = df.iloc[val_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        
        logger.info(f"Time-based split:")
        logger.info(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
        logger.info(f"  Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()}")
        logger.info(f"  Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()}")
        
        return train_df, val_df, test_df
    
    def _prepare_data(self, df: pd.DataFrame, feature_columns: list) -> tuple:
        X = df[feature_columns].copy()
        y = df['target_net_gbp'].copy()
        
        return X, y
    
    def _build_model(self) -> Pipeline:
        model_type = self.model_config['type']
        hyperparams = self.model_config['hyperparameters']
        
        # Select model
        if model_type == 'RandomForestRegressor':
            regressor = RandomForestRegressor(**hyperparams)
        elif model_type == 'GradientBoostingRegressor':
            regressor = GradientBoostingRegressor(**hyperparams)
        elif model_type == 'LinearRegression':
            regressor = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Build pipeline with imputer
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', regressor)
        ])
        
        logger.info(f"Built model pipeline: {model_type}")
        
        return pipeline
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, set_name: str) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (handle zero values)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'n_samples': len(y_true),
            'mean_actual': float(np.mean(y_true)),
            'mean_predicted': float(np.mean(y_pred))
        }
        
        logger.info(f"\n{set_name} Metrics:")
        logger.info(f"  MAE:  £{mae:.2f}")
        logger.info(f"  RMSE: £{rmse:.2f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def _log_feature_importance(self, top_n: int = 20):
        try:
            importances = self.model.named_steps['regressor'].feature_importances_
            
            # Get feature names after imputation (some may have been removed)
            # The imputer outputs the same features, but we need to match lengths
            if len(importances) != len(self.feature_columns):
                logger.warning(f"Feature importance length ({len(importances)}) != feature columns length ({len(self.feature_columns)})")
                # Use a subset or generate generic names
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                feature_names = self.feature_columns
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop {top_n} Feature Importances:")
            for idx, row in feature_importance.head(top_n).iterrows():
                logger.info(f"  {row['feature']:<40s} {row['importance']:.4f}")
            
            # Store for saving
            self.feature_importance = feature_importance
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")
    
    def save(self, model_dir: str):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model pipeline
        model_path = model_dir / 'model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save feature columns
        features_path = model_dir / 'feature_columns.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info(f"Saved feature columns to {features_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_config['type'],
            'hyperparameters': self.model_config['hyperparameters'],
            'training_date': datetime.now().isoformat(),
            'metrics': self.metrics,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save feature importance if available
        if hasattr(self, 'feature_importance'):
            fi_path = model_dir / 'feature_importance.csv'
            self.feature_importance.to_csv(fi_path, index=False)
            logger.info(f"Saved feature importance to {fi_path}")
        
        logger.info(f"Model artifacts saved to {model_dir}")


class ModelLoader:
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_columns = None
        self.metadata = None
        
        self._load()
    
    def _load(self):
        # Load model
        model_path = self.model_dir / 'model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load feature columns
        features_path = self.model_dir / 'feature_columns.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        # Load metadata
        metadata_path = self.model_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Loaded model metadata")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Ensure features are in correct order
        X = X[self.feature_columns]
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_metadata(self) -> dict:
        return self.metadata


def train_model(features_df: pd.DataFrame, feature_columns: list, config: dict) -> ModelTrainer:
    trainer = ModelTrainer(config)
    trainer.train(features_df, feature_columns)
    return trainer


def load_model(model_dir: str) -> ModelLoader:
    return ModelLoader(model_dir)

