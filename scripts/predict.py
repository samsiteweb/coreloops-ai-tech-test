"""CLI prediction script."""

import sys
import logging
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.aggregator import load_aggregated_metrics
from src.features.engineer import FeatureEngineer
from src.model.predict import Predictor

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Less verbose for CLI
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Predict next-day customer spending'
    )
    
    parser.add_argument(
        '--customer',
        type=str,
        required=True,
        help='Customer ID (e.g., C00042)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Target date in YYYY-MM-DD format (e.g., 2024-10-06)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Load config
    config = load_config()
    
    # Load aggregated metrics
    metrics_file = config['output']['metrics_file']
    
    if not Path(metrics_file).exists():
        print(f"ERROR: Metrics file not found: {metrics_file}")
        print("Run the pipeline first: python scripts/run_pipeline.py")
        sys.exit(1)
    
    metrics_df = load_aggregated_metrics(metrics_file)
    
    # Check if model exists
    model_dir = config['output']['model_dir']
    model_file = Path(model_dir) / 'model.pkl'
    
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        print("Train the model first: python scripts/train_model.py")
        sys.exit(1)
    
    predictor = Predictor(model_dir, metrics_df)
    engineer = FeatureEngineer(config)
    
    # Make prediction
    try:
        result = predictor.predict_for_customer_date(
            customer_id=args.customer,
            target_date=args.date,
            feature_engineer=engineer
        )
        
        # Output result
        print(f"\nCustomer {result['customer_id']} on {result['target_date']}")
        
        if 'error' in result:
            print(f"Warning: {result['error']}")
            print(f"Predicted: £{result['prediction']:.2f}")
        else:
            print(f"Predicted: £{result['prediction']:.2f} ({result['confidence']} confidence)")
            print(f"Based on {result['history_days']} days of history, {result['recent_transactions_30d']} recent transactions")
        
    except Exception as e:
        print(f"\nERROR: Prediction failed: {e}")
        if args.verbose:
            logger.error("Prediction error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()




