# Coreloops – Take-home - AI Engineer

**Estimated time:** ~4 hours  
**Language:** Python preferred (TypeScript/Node.js also acceptable).  
**Package manager:** If using Node.js, please use **pnpm**.  
**You may NOT use external paid services.**

---

## Scenario

You receive **daily CSV files** of customer transactions (one file per day).  
The files contain mixed currencies (GBP/EUR/USD), occasional missing values, product returns (negative quantities), and some duplicate rows.

Your task is to build a **small, production-minded data + ML pipeline** that:

1. **Loads** all available daily files from `data/` (assume new files appear over time).
2. **Normalises** the data:
    - Deduplicate rows (define your criteria).
    - Handle missing `customer_id`, `unit_price`, and `description` sensibly.
    - Convert all monetary values to **GBP** using `fx_rates.csv`.
3. **Aggregates** the data into **daily per-customer metrics**, including at minimum:
    - `date`
    - `customer_id`
    - `orders` – count of distinct invoices
    - `items` – sum of absolute quantities
    - `gross_gbp` – sum of `quantity * unit_price` in GBP for positive quantities
    - `returns_gbp` – sum for negative quantities
    - `net_gbp` – gross + returns
4. **Trains** a simple model to **predict next-day `net_gbp` per customer**.
    - You decide which features to use (e.g. rolling averages, category proportions, recency).
    - Provide a clear **train/validation** strategy (time-based split preferred).
    - Evaluate using an appropriate metric (e.g. MAE or RMSE).
5. **Outputs**:
    - `artifacts/daily_customer_metrics.parquet` (or `.csv`)
    - `artifacts/model/` containing saved model parameters or reproducible configuration
    - A small CLI command, for example:
      ```bash
      pnpm run predict -- --customer C00042 --date 2024-10-06
      ```
      which returns a predicted `net_gbp` value for that date.

---

## Data

Data lives in a **public Google Cloud Storage bucket**:  
`gs://tech-test-file-storage/`

Accessible via HTTPS as: `https://storage.googleapis.com/tech-test-file-storage/`

Structure:

- `data/*` – daily files to be used for ingestion
- `fx_rates.csv` – the daily FX conversion rates

**Example URL:**
`https://storage.googleapis.com/tech-test-file-storage/data/2024-10-01.csv`

---

## Deliverables

A single repo or folder containing:

- `src/` – ETL, transformation, and feature engineering code
- `scripts/` – CLI entry points
- `README.md` – clear run instructions, assumptions, and design notes
- *(Optional)* lightweight unit or data validation tests

---

## Evaluation Criteria

| Area | Focus                                                                                                                      |
|------|----------------------------------------------------------------------------------------------------------------------------|
| **Data Engineering** | Clean, reproducible ingestion and transformation logic; well-documented deduplication and null handling                    |
| **Aggregation & Features** | Sound daily metrics, clear business reasoning, good feature design                                                         |
| **ML Component** | Sensible validation, explainable model choice, reproducibility                                                             |
| **Code Quality** | Readable, modular, and runnable end-to-end with clear entry points                                                         |
| **Communication** | Clear reasoning for design choices and trade-offs, with clear documentation on how to run and setup the project from fresh |

---

## Hints

- Treat **negative quantities** as returns.
- If `unit_price` is missing, you may impute from the product’s median price (per currency/day) or drop the row — justify your approach.
- Make FX conversion explicit and traceable.
- Prioritise clarity, reproducibility, and correctness over framework sophistication.
