# CreditGuard — Step-by-Step Build & Run Guide

For each step: **what** · **why** · **run** · **expected output** · **check before moving on**.
Run from the project root with the virtual environment (`.venv\Scripts\python.exe`, or activate it first).

Mapped to the course milestones: Steps 0–2 support **Milestone 1** (problem, dataset, methodology — 5–9 Oct),
Steps 3–8 are **Milestone 2** (implementation + demo — 26–29 Oct), Step 9 is **Milestone 3** (final report, 16–20 Nov).

---

## Step 0 — Environment
- **What:** Python 3.12 venv, Java 17, `C:\hadoop\bin\winutils.exe` + `hadoop.dll`, Node.js.
- **Why:** Spark runs on the JVM; on Windows it needs winutils to write files.
- **Run:** `pip install -r backend/requirements.txt` · `cd frontend && npm install`
- **Check:** `python -c "import pyspark; print(pyspark.__version__)"` → `3.5.9`; `java -version` → 17.

## Step 1 — Ingestion & cleaning (`s01_ingest_clean.py`)
- **What:** Load both CSVs with explicit schemas, check quality, clean, drop personal columns, save Parquet.
- **Expected:** ULB 284,807 → 283,726 rows (473 fraud); Sparkov 1,852,394 rows (9,651 fraud); ~40 s.
- **Check:** `data/processed/ulb_clean.parquet` and `sparkov_clean.parquet` exist.

## Step 2 — EDA & statistics (`s02_eda.py`)
- **What:** Spark SQL aggregations, Welch t-test, chi-square test, 11 EDA charts.
- **Expected:** "shopping_net has the highest fraud rate (1.59%)", "Fraud concentrates at hours [0, 1, 22, 23]".
- **Check:** `reports/figures/eda_*.png` (11 files) and `reports/results/eda_stats.json`.

## Step 3 — Feature engineering (`s03_features.py`)
- **What:** Window functions (card history), time, age, distance, log features.
- **Expected:** table showing fraud `amt_to_avg_ratio` 7.06 vs 0.98, `is_night` 0.85 vs 0.23.
- **Check:** `data/processed/*_features.parquet`.

## Step 4 — Train 5 models × 2 datasets (`s04_train_models.py`) — ≈10 min
- **What:** LR, Decision Tree, Random Forest, GBT, Linear SVM; class weights; threshold tuned on validation; best by
  validation F1; saved as Spark PipelineModel.
- **Expected:** "SELECTED: Random Forest" for both datasets; Sparkov RF test F1 ≈ 0.86, ULB RF ≈ 0.84.
- **Check:** `models/spark/ulb_best/`, `models/spark/sparkov_best/`, `reports/results/model_comparison.csv`.

## Step 4b — Demo samples (`s04b_demo_samples.py`)
- **Expected:** "ulb: 6 legit + 6 fraud samples", same for sparkov → `models/spark/samples.json`.

## Step 5 — Evaluation charts (`s05_evaluate.py`)
- **What:** comparison bars, confusion matrices, ROC + PR curves, feature importance, training time.
- **Check:** 11 more PNGs in `reports/figures/` (21 total).

## Step 6 — Website (`start.bat`)
- **What:** FastAPI loads both Spark models; React UI with Predict (2 dataset tabs), Result, Dashboard, History, About.
- **Check:** top-right shows **API online** (after ~20 s); load a suspicious Dataset 2 sample → "⚠ Potential Fraud
  Detected"; `cd backend && pytest -v` → **9 passed**.

## Step 7 — Real-time streaming (`start_streaming_demo.bat`)
- **What:** producer sends micro-batches; Spark Structured Streaming scores them, prints ALERT lines, writes to SQLite.
- **Check:** streaming window shows `[batch N] 50 transactions scored | k fraud alert(s)`; website Dashboard with
  **Live refresh** shows the "Real-Time Stream" count rising.

## Step 8 — Power BI (`s07_export_powerbi.py` + `powerbi/POWERBI_GUIDE.md`)
- **What:** export `fact_transactions.csv` (1.85M rows) + model tables; build 4–5 report pages in Power BI Desktop.
- **Check:** `powerbi/data/` has the CSVs; report saved as `powerbi/CreditGuard_Dashboard.pbix`.

## Step 9 — Final submission checklist (LMS, by 20 Nov 2026)
| Item | Where |
|---|---|
| Report chapters 1–5 (your own words, Turnitin < 10%, AI 0%) | you write it |
| All dataset files | `data/creditcard.csv`, `data/sparkov/*.csv` (+ links in `data/README.md`) |
| Complete source code + README | this folder (without `.venv`, `node_modules`) |
| Screenshots of results | `reports/figures/`, `docs/screenshots/`, Power BI pages |
| Plagiarism report | from Turnitin |
