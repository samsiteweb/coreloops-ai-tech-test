# Customer Spending Predictor

Predicts how much a customer will spend tomorrow based on their transaction history.

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

Run these three commands in order:

```bash
# 1. Process the data
python3 scripts/run_pipeline.py

# 2. Train the model
python3 scripts/train_model.py

# 3. Make a prediction
python3 -m scripts.predict --customer C00042 --date 2024-10-06
```

Done! You'll see the predicted spending amount.

## Try Different Customers

```bash
python3 -m scripts.predict --customer C00100 --date 2024-10-06
python3 -m scripts.predict --customer C00042 --date 2024-10-06 --verbose
```

## What happens

Step 1 downloads and cleans transaction data, converts everything to GBP. Step 2 trains a RandomForest model. Step 3 makes predictions for any customer.

## Troubleshooting

If you get "module not found" errors, run `pip install -r requirements.txt`

If you get "model not found", make sure you ran steps 1 and 2 first.

## Optional: Explore the Data

```bash
python3 scripts/explore_data.py
```

Shows stats about transactions, duplicates, missing values, etc.

