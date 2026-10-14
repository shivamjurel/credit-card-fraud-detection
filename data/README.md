# Datasets

Both files are too large for GitHub and are ignored by Git. Download them into this folder.

## Dataset 1 — ULB European Credit Card Fraud (`data/creditcard.csv`)

- **Source:** Machine Learning Group, Université Libre de Bruxelles —
  Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Direct mirror (no login):** https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- **Size:** ~144 MB · 284,807 transactions · 492 frauds (0.172%) · September 2013, 2 days, European cardholders
- **Columns:** `Time` (seconds since first transaction), `V1`–`V28` (PCA-anonymised), `Amount` (EUR), `Class` (1 = fraud)

## Dataset 2 — Sparkov Simulated Card Transactions (`data/sparkov/fraudTrain.csv`, `fraudTest.csv`)

- **Source:** Kartik Shenoy — Kaggle "Credit Card Transactions Fraud Detection Dataset":
  https://www.kaggle.com/datasets/kartik2112/fraud-detection (generated with the Sparkov data generator)
- **Download:** Kaggle page → Download (zip, ~202 MB), or
  `curl -L -o archive.zip https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection`
  then unzip into `data/sparkov/`
- **Size:** ~478 MB unzipped · 1,852,394 transactions · 9,651 frauds (0.52%) · Jan 2019 – Dec 2020, US customers
  - `fraudTrain.csv`: 1,296,675 rows (Jan 2019 – Jun 2020)
  - `fraudTest.csv`: 555,719 rows (Jun 2020 – Dec 2020)
- **Columns:** `trans_date_trans_time, cc_num, merchant, category, amt (USD), first, last, gender, street, city, state,
  zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud`
- The data is **simulated** (no real people). Names, street, zip and transaction number are still dropped by the
  pipeline (data minimisation), and the card number is replaced by an anonymous hash.

## Generated folders (created by `spark_pipeline/`)

- `data/processed/` — cleaned + feature-engineered Parquet files
- `data/stream/` — streaming demo input, scored output and checkpoint
