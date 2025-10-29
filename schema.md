# Transactions schema

Each CSV in `data/` represents one day of transactions.

| column           | type        | notes |
|------------------|-------------|-------|
| invoice_id       | string      | invoice identifier; multiple rows per invoice (one per product) |
| customer_id      | string?     | may be missing in ~2% rows (simulate dirty data) |
| country          | string      | ISO country code (GB, US, DE, FR, IE, ES, IT) |
| currency         | string      | One of GBP, USD, EUR (derived from country) |
| product_id       | string      | product SKU (P####) |
| product_category | string      | categorical |
| description      | string?     | may be NULL / missing |
| quantity         | integer     | negative values indicate returns |
| unit_price       | float?      | price in the row's currency; occasionally NULL |
| timestamp        | ISO datetime| purchase time within the day |

## FX rates

`fx_rates.csv` provides daily conversion rates **to GBP**.

| column   | type   |
|----------|--------|
| date     | YYYY-MM-DD |
| currency | GBP, USD, EUR |
| rate_to_gbp | float (GBP=1.0) |
