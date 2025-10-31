"""Model training script."""

import sys
import logging
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.aggregator import load_aggregated_metrics
from src.features.engineer import FeatureEngineer
from src.model.train import ModelTrainer

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
    logger.info("Training model...")
    
    config = load_config()
    
    logger.info("Loading metrics...")
    metrics_file = config['output']['metrics_file']
    metrics_df = load_aggregated_metrics(metrics_file)
    logger.info(f"Loaded {len(metrics_df):,} records")
    
    logger.info("\nEngineering features...")
    engineer = FeatureEngineer(config)
    features_df = engineer.create_features(metrics_df)
    feature_columns = engineer.get_feature_columns(features_df)
    logger.info(f"Created {len(feature_columns)} features")
    
    logger.info("\nTraining...")
    trainer = ModelTrainer(config)
    metrics = trainer.train(features_df, feature_columns)
    
    logger.info("\nSaving model...")
    model_dir = config['output']['model_dir']
    trainer.save(model_dir)
    
    logger.info(f"\nValidation MAE: £{metrics['validation']['mae']:.2f}, RMSE: £{metrics['validation']['rmse']:.2f}, R²: {metrics['validation']['r2']:.4f}")
    logger.info(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)




