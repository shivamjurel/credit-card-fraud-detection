# CreditGuard — Viva Preparation Guide (CSET-371 Big Data)

Simple English answers with short Hinglish hints (💬). All numbers come from **this** project's actual runs.
Use this to *understand* and *speak* — write your report in your own words (the course requires 0% AI plagiarism).

---

## Part A — Big Data & Apache Spark

### 1. Why is this a Big Data problem?
The **3 Vs**: *Volume* – 2.1 million transactions (≈620 MB raw CSV) here, billions per day in real banks;
*Velocity* – transactions arrive continuously and fraud must be caught in seconds (→ streaming);
*Variety* – anonymised numeric (PCA), categorical (merchant category), time, geo-location, customer data.
💬 *Data bahut zyada, bahut tez, aur alag-alag type ka hai — isliye normal Excel/Pandas se scale nahi hoga.*

### 2. What is Apache Spark?
An open-source **distributed computing engine**. It splits data into **partitions** and processes them in parallel on
many CPU cores or many machines, keeping data **in memory** (much faster than Hadoop MapReduce, which writes to disk
between steps). We use **PySpark** (Python API): Spark SQL for queries, MLlib for ML, Structured Streaming for real time.
💬 *Ek bada kaam chhote tukdon mein baant ke saath-saath karna.*

### 3. Spark architecture?
**Driver** (our Python program, builds the plan) → **Cluster manager** → **Executors** (do the work on partitions).
Here we run `master("local[*]")`: driver and executors on one PC using all 24 cores. The same code runs on a cluster
or Databricks by changing only the master URL — that is **scalability**.

### 4. RDD vs DataFrame?
RDD = low-level distributed collection. **DataFrame** = distributed table with a schema; Spark's **Catalyst optimizer**
plans the most efficient execution. We use DataFrames everywhere.

### 5. What is lazy evaluation? Transformations vs actions?
**Transformations** (`filter`, `withColumn`, `groupBy`) only build a plan (DAG). Nothing runs until an **action**
(`count`, `collect`, `write`, `show`). This lets Spark optimise the whole chain.
💬 *Spark pehle poora plan banata hai, kaam tab karta hai jab result maanga jaata hai.*

### 6. Why Parquet instead of CSV?
Parquet is **columnar + compressed + stores the schema**. Reading only needed columns is fast; files are smaller.
Standard format in Big Data lakes. (Our cleaned tables *and* feature tables for both datasets together take only
≈365 MB of Parquet, vs ≈620 MB for the raw CSVs alone.)

### 7. What does `cache()` do?
Keeps a DataFrame in memory after first use, so repeated use (training 5 models on the same data) does not re-read
and recompute it.

### 8. What are window functions? Where did you use them?
Calculations over a group of related rows **without collapsing** them. We partition by card and order by time:
`count(*) over (last 24 h)` → transactions in the last 24 hours, `avg(amt) over (all earlier rows)` → card's usual
amount, `lag(ts)` → time since the previous transaction. Only *past* rows are used → **no data leakage**.
💬 *Har transaction ke liye usi card ki pichli history dekhna.*

### 9. What is Spark MLlib? What is a Pipeline?
Spark's distributed ML library. A **Pipeline** chains stages: `StringIndexer → OneHotEncoder → VectorAssembler →
StandardScaler → Classifier`. Calling `fit()` trains all stages; the resulting **PipelineModel** is saved and reused
by the website and the streaming job, so preprocessing is always identical.

### 10. What is Spark Structured Streaming? How does your demo work?
It treats a live stream as an **unbounded table** and processes it in **micro-batches**. Our producer drops a JSON file
of 50 real test transactions every 2 s; `readStream` picks each file up, the saved model scores it, and
`foreachBatch` prints alerts and writes to Parquet + SQLite (website updates live). A **checkpoint** remembers
processed files → fault tolerance (no duplicates after restart). Demo result: 500 transactions, 12 of 15 frauds
caught, 0 false alarms.

### 11. What is Kafka and how would it fit?
A distributed message queue for real-time event streams. In production the producer would publish transactions to a
Kafka topic and Spark would read it with `readStream.format("kafka")` — a one-line source change.

### 12. Hadoop / HDFS / Hive — did you use them?
Spark on Windows uses Hadoop's file-system libraries (that's why `winutils.exe` is needed). HDFS (distributed storage)
and Hive (SQL tables over files) are listed as future scope — the PDF allows them only as supplementary components.

### 13. What is Power BI's role?
Business Intelligence / presentation layer. Spark does heavy processing and exports one clean fact table (1.85M rows)
+ model results; Power BI builds interactive KPIs, trends, maps and model-performance pages for decision makers.

### 14. Why did you use two datasets?
Dataset 1 (ULB) is **real** but anonymised (V1–V28 cannot be explained). Dataset 2 (Sparkov) is **larger, readable and
time-ordered**, enabling behavioural features, a realistic time-based test and meaningful Power BI insights.
Comparing both shows the models generalise.

---

## Part B — Machine Learning

### 15. What is classification / supervised learning?
Predicting a **category** (Fraud / Legitimate) from labelled examples (each training row has the correct answer).
💬 *Teacher (label) ke saath padhai.*

### 16. What is an imbalanced dataset? How did you handle it?
Fraud is 1 in 598 (ULB) and 1 in 190 (Sparkov). We used **class weights** (`weightCol`): each fraud row counts ≈87×
(Sparkov) / ≈308× (ULB) more, so models cannot ignore fraud. We also **tuned the decision threshold**.

### 17. Why is accuracy insufficient?
A model that always says "Legitimate" is 99.83% (ULB) / 99.48% (Sparkov) accurate on the full datasets but catches
**zero** fraud.

### 18. Precision, Recall, F1?
- **Precision** = TP / (TP + FP) — of flagged transactions, how many were fraud. RF Sparkov: 1,747 / (1,747 + 169) = 91.2%.
- **Recall** = TP / (TP + FN) — of real frauds, how many we caught: 1,747 / 2,145 = 81.5%.
- **F1** = harmonic mean = 0.860. High only when both are good → our selection metric.

### 19. ROC-AUC vs PR-AUC?
ROC-AUC measures ranking across all thresholds (TPR vs FPR). With rare fraud, FPR stays tiny even with many false
alarms, so **every** model scored ≈0.99. **PR-AUC** (precision vs recall) exposes the difference: 0.29 (SVM) vs 0.91
(RF). 💬 *Rare class ke liye PR-AUC zyada imaandaar metric hai.*

### 20. Confusion matrix (RF, Sparkov test)?
|  | Pred Legit | Pred Fraud |
|---|---|---|
| **Actual Legit** | TN 553,405 | FP 169 |
| **Actual Fraud** | FN 398 | TP 1,747 |
FN (missed fraud) costs money; FP annoys customers.

### 21. Explain your 5 models in one line each.
- **Logistic Regression** – weighted sum → sigmoid → probability; linear baseline.
- **Decision Tree** – if/else splits on features; explainable, can overfit.
- **Random Forest** – 100 trees on random rows/features, averaged vote; robust. **Selected on both datasets.**
- **Gradient-Boosted Trees** – trees built one after another, each fixing previous errors; best recall (86%) but slower (145 s).
- **Linear SVM** – widest-margin linear boundary; no probabilities (we squash the margin with a sigmoid in the API).

### 22. Why did tree models win?
Fraud depends on **interactions** (large amount *and* night *and* unusual for this card). Trees capture non-linear
interactions; linear models cannot — hence F1 0.86 vs 0.47–0.49.

### 23. Train / validation / test — why three sets?
Train = learn; **validation = choose threshold and model**; test = final unbiased score (used once). Choosing on the
test set would be cheating (optimistic results). Sparkov uses a **time-based** split: train on 2019–mid 2020, test on
later months — like the real world.

### 24. Why scale features? StandardScaler?
z = (x − mean) / std → mean 0, std 1. Needed for LR and SVM (range of `amt` or `city_pop` would dominate); trees don't
need it but share the same pipeline.

### 25. Overfitting / underfitting?
Overfitting = memorises training data, fails on new data (limited with `maxDepth=10`, `minInstancesPerNode`, many
trees, and checked with the unseen test set). Underfitting = too simple (our linear models on Sparkov).

### 26. Most important features?
Sparkov RF: `amt`, `amt_sum_24h`, `log_amt`, `amt_to_avg_ratio`, `is_night`, `hour` — amount and card behaviour.
ULB: V14, V4, V17, V10, V12.

---

## Part C — Application

### 27. How does a website prediction flow?
React form → Axios `POST /predict/sparkov` → Pydantic validation → `predictor.py` builds features → 1-row Spark
DataFrame → saved **PipelineModel** → score → Fraud if ≥ tuned threshold (0.899) → risk level → saved in SQLite →
JSON → Result page. History and Dashboard read the same SQLite table.

### 28. How is the Spark model loaded in FastAPI?
At startup (`lifespan`) the backend starts a small local SparkSession and `PipelineModel.load()`s both models once;
every request reuses them. If Java/models are missing, `/predict` returns **503**.

### 29. Why FastAPI? Why SQLite?
FastAPI: Python (same as Spark code), automatic validation + Swagger docs. SQLite: single file, zero setup — fine for
storing demo predictions; production would use PostgreSQL or a data lake.

### 30. Risk levels?
High = score ≥ the model's tuned threshold (= predicted fraud). Medium = ≥ half the threshold. Low otherwise.
Demonstration settings, not banking rules.

### 31. SDGs?
SDG 8 (target 8.10 – stronger financial institutions), SDG 9 (innovation & infrastructure), SDG 16 (target 16.4 –
reduce illicit financial flows).

### 32. Limitations?
Simulated Dataset 2; local-mode Spark; recall 81% (≈1 in 5 frauds missed); streaming uses pre-computed card-history
features; thresholds are demo values.
