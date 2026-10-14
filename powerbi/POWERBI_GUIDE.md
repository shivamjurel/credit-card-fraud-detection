# CreditGuard — Power BI Dashboard Guide

The Spark pipeline prepares the data (`spark_pipeline/s07_export_powerbi.py`); **Power BI Desktop** turns it into an
interactive Business Intelligence dashboard. Follow these steps once (≈45–60 minutes) and save the result as
`powerbi/CreditGuard_Dashboard.pbix`.

---

## 0. Install Power BI Desktop (free)

Microsoft Store → search **"Power BI Desktop"** → Install. (Windows only; no account needed for Desktop.)

## 1. Make sure the data exists

```bash
python spark_pipeline/s07_export_powerbi.py
```

Folder `powerbi/data/` should contain:

| File | Rows | What it is |
|---|---|---|
| `fact_transactions.csv` | 1,852,394 | Every Sparkov transaction with business columns + model score for the test period |
| `model_comparison.csv` | 10 | 5 models × 2 datasets, all metrics |
| `feature_importance.csv` | 30 | Top 15 features of each selected model |
| `ulb_hourly.csv` | 24 | Dataset 1 transactions & fraud per hour |
| `stream_scored.csv` | varies | Transactions scored by the real-time stream (only if you ran the streaming demo before exporting) |

**`fact_transactions` columns:** `transaction_id, transaction_datetime, transaction_date, hour, day_name, year_month,
category, merchant, amount, gender, age, age_group, job, city, state, lat, long, city_pop, distance_km, txn_count_24h,
amount_to_card_avg, is_night, is_fraud, data_split, fraud_score, predicted_fraud, card_id`
(`fraud_score` / `predicted_fraud` are filled only for `data_split = Test`.)

## 2. Load the data

1. Open Power BI Desktop → **Home → Get data → Text/CSV**.
2. Select `powerbi/data/fact_transactions.csv` → **Transform Data** (opens Power Query).
3. Check data types (click the icon left of each column name):
   - `transaction_datetime` → Date/Time, `transaction_date` → Date
   - `amount, distance_km, amount_to_card_avg, fraud_score, lat, long` → Decimal number
   - `hour, age, city_pop, txn_count_24h, is_fraud, is_night, predicted_fraud` → Whole number
   - If decimals look wrong (e.g. 6389 instead of 63.89): **File → Options → Regional settings → Locale = English (United States)**.
4. Rename the query to `fact_transactions` → **Close & Apply** (loading 1.85M rows takes ~1–2 min).
5. Repeat **Get data → Text/CSV** for `model_comparison.csv`, `feature_importance.csv`, `ulb_hourly.csv`
   (and `stream_scored.csv` if present). These tables are independent — no relationships needed.
6. Column tools: select `state` → **Data category = State or Province**; `city` → City; `lat` → Latitude;
   `long` → Longitude. Select `day_name` → **Sort by column** needs a helper: add a column
   `day_number = WEEKDAY(fact_transactions[transaction_date], 2)` and sort `day_name` by it.

## 3. Create DAX measures

**Home → New measure** (in table `fact_transactions`), one at a time:

```DAX
Total Transactions = COUNTROWS ( fact_transactions )

Fraud Transactions = CALCULATE ( [Total Transactions], fact_transactions[is_fraud] = 1 )

Fraud Rate % = DIVIDE ( [Fraud Transactions], [Total Transactions] )

Total Amount = SUM ( fact_transactions[amount] )

Fraud Amount = CALCULATE ( [Total Amount], fact_transactions[is_fraud] = 1 )

Avg Fraud Amount = DIVIDE ( [Fraud Amount], [Fraud Transactions] )

Avg Legit Amount = CALCULATE ( AVERAGE ( fact_transactions[amount] ), fact_transactions[is_fraud] = 0 )

Test Frauds = CALCULATE ( [Fraud Transactions], fact_transactions[data_split] = "Test" )

Frauds Caught = CALCULATE ( [Total Transactions], fact_transactions[is_fraud] = 1, fact_transactions[predicted_fraud] = 1 )

False Alarms = CALCULATE ( [Total Transactions], fact_transactions[is_fraud] = 0, fact_transactions[predicted_fraud] = 1 )

Model Recall % = DIVIDE ( [Frauds Caught], [Test Frauds] )

Model Precision % = DIVIDE ( [Frauds Caught], [Frauds Caught] + [False Alarms] )

Fraud Amount Detected = CALCULATE ( [Total Amount], fact_transactions[is_fraud] = 1, fact_transactions[predicted_fraud] = 1 )
```

Format `Fraud Rate %`, `Model Recall %`, `Model Precision %` as **Percentage** (Measure tools → %), and the amount
measures as **Currency ($)**.

## 4. Build the report pages

Use a consistent colour rule: **red = fraud, green = legitimate, blue = neutral** (same as the website).

### Page 1 — Executive Overview
| Visual | Fields |
|---|---|
| 4 × **Card** | Total Transactions · Fraud Transactions · Fraud Rate % · Fraud Amount |
| **Line chart** | X: `year_month` · Y: Fraud Transactions (title "Monthly fraud trend") |
| **Donut chart** | Legend: `is_fraud` · Values: Total Amount |
| **Slicers** | `year_month` (between), `category` (dropdown), `state` (dropdown), `data_split` |

### Page 2 — Fraud Patterns (EDA)
| Visual | Fields |
|---|---|
| **Clustered bar** | Y: `category` · X: Fraud Rate % (sort descending) |
| **Clustered column** | X: `hour` · Y: Fraud Rate % → shows the late-night peak (22:00–03:00) |
| **Matrix (heatmap)** | Rows: `day_name` · Columns: `hour` · Values: Fraud Transactions → Conditional formatting → Background colour (white → red) |
| **Clustered column** | X: `age_group` · Y: Fraud Rate % |
| **Card pair** | Avg Fraud Amount vs Avg Legit Amount (≈ $531 vs $68) |

### Page 3 — Geography
> First enable maps once: **File → Options → Security → "Use Map and Filled Map visuals"**.

| Visual | Fields |
|---|---|
| **Map** (bubble) | Latitude: `lat` · Longitude: `long` · Bubble size: Fraud Transactions · Tooltips: `city`, Fraud Rate % |
| **Filled map** | Location: `state` · Colour saturation: Fraud Rate % |
| **Table** | `state`, `city`, Fraud Transactions, Fraud Amount — top 10 by Fraud Transactions |

### Page 4 — Model Performance (Spark MLlib)
| Visual | Table / Fields |
|---|---|
| **Clustered bar** | `model_comparison`: Y `model` · X `f1_score` · Legend `dataset` |
| **Table** | `model_comparison`: dataset, model, precision, recall, f1_score, roc_auc, pr_auc, train_seconds (conditional formatting: data bars on f1_score) |
| **Clustered bar** | `feature_importance`: Y `feature` · X `importance` · Filter: dataset = "sparkov" |
| **Cards** | Model Recall % · Model Precision % · Frauds Caught · False Alarms · Fraud Amount Detected (filter `data_split = Test`) |

### Page 5 — Real-Time Stream (optional, if `stream_scored.csv` exists)
| Visual | Fields |
|---|---|
| **Cards** | Count of rows · Sum of `predicted_fraud` |
| **Table** | event_time, city, state, category, amt, score, predicted_fraud (filter predicted_fraud = 1) — "fraud alerts" |

## 5. Finish

1. **View → Themes** → pick a clean theme (e.g. "Executive"); add the title "CreditGuard – Fraud Analytics" on every page.
2. Save as `powerbi/CreditGuard_Dashboard.pbix`.
3. Take a screenshot of every page for the report (`docs/screenshots/powerbi_*.png`).
4. (Optional) **Publish** to Power BI Service if your college account allows it.

## 6. What to say in the viva

- *"Spark does the heavy processing (1.85M rows: cleaning, feature engineering, model scoring) and writes one clean
  table. Power BI is the presentation layer: it lets a business user filter by month, category or state and see KPIs
  instantly."*
- *"The Model Performance page compares all five Spark MLlib models and shows, on unseen test data, how many frauds
  the selected Random Forest caught and how much money that represents."*
