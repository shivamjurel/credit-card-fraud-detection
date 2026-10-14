"""
Step 2 - Exploratory Data Analysis & Statistical Analysis (PySpark)
===================================================================

Run:  python spark_pipeline/s02_eda.py

All aggregations run in Spark (distributed). Only the small aggregated results
are brought to the driver to draw charts with matplotlib.

Outputs:
  reports/figures/eda_*.png        charts for the report / presentation
  reports/results/eda_stats.json   numbers + statistical test results
"""

import json
import math
import time

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import functions as F

import config
from plot_style import FRAUD_COLOR, LEGIT_COLOR, MAIN_COLOR, plt, save
from spark_session import get_spark

FIG = config.FIGURES_DIR
LABEL = config.LABEL


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def welch_t_test(df, column: str) -> dict:
    """
    Welch's t-test: is the mean of `column` different for fraud vs legitimate?
    Mean, variance and count are computed in Spark; the formula is applied on the driver.
    """
    stats = {
        row[LABEL]: row
        for row in df.groupBy(LABEL)
        .agg(F.mean(column).alias("mean"), F.variance(column).alias("var"), F.count("*").alias("n"))
        .collect()
    }
    legit, fraud = stats[0], stats[1]
    standard_error = math.sqrt(legit["var"] / legit["n"] + fraud["var"] / fraud["n"])
    t_value = (fraud["mean"] - legit["mean"]) / standard_error
    # Normal approximation of the p-value (n is very large, so t ~ normal)
    p_value = math.erfc(abs(t_value) / math.sqrt(2))
    return {
        "mean_legit": round(legit["mean"], 4),
        "mean_fraud": round(fraud["mean"], 4),
        "t_statistic": round(t_value, 3),
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
    }


def class_distribution_chart(counts: dict, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Legitimate", "Fraud"]
    values = [counts[0], counts[1]]
    bars = ax.bar(labels, values, color=[LEGIT_COLOR, FRAUD_COLOR], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(title)
    total = sum(values)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:,}\n({value / total:.2%})", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.25)
    save(fig, path)


# --------------------------------------------------------------------------
# Dataset 1 - ULB
# --------------------------------------------------------------------------
def eda_ulb(spark) -> dict:
    section("EDA - DATASET 1 (ULB)")
    df = spark.read.parquet(config.spark_path(config.ULB_CLEAN)).cache()
    results = {}

    counts = {row[LABEL]: row["count"] for row in df.groupBy(LABEL).count().collect()}
    results["class_counts"] = {"legitimate": counts[0], "fraud": counts[1]}
    print(f"  Legitimate {counts[0]:,} | Fraud {counts[1]:,} | ratio 1:{counts[0] // counts[1]}")
    class_distribution_chart(counts, "ULB - Class distribution", FIG / "eda_ulb_class_distribution.png")

    # Summary statistics of Amount per class
    amount_stats = (
        df.groupBy(LABEL)
        .agg(
            F.mean("Amount").alias("mean"),
            F.expr("percentile_approx(Amount, 0.5)").alias("median"),
            F.max("Amount").alias("max"),
        )
        .orderBy(LABEL)
        .collect()
    )
    results["amount_by_class"] = {("fraud" if r[LABEL] else "legitimate"): {k: round(r[k], 2) for k in ["mean", "median", "max"]} for r in amount_stats}
    print(f"  Amount by class: {results['amount_by_class']}")
    results["amount_t_test"] = welch_t_test(df, "Amount")
    print(f"  Welch t-test on Amount: {results['amount_t_test']}")

    # Fraud rate by hour of day (Time is seconds from the first transaction)
    hourly = (
        df.withColumn("hour", (F.floor(F.col("Time") / 3600) % 24).cast("int"))
        .groupBy("hour")
        .agg(F.count("*").alias("n"), F.sum(LABEL).alias("fraud"))
        .orderBy("hour")
        .toPandas()
    )
    hourly["fraud_rate_pct"] = hourly["fraud"] / hourly["n"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(hourly["hour"], hourly["fraud_rate_pct"], color=MAIN_COLOR, width=0.7)
    ax.set_xlabel("Hour of day (from start of data collection)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("ULB - Fraud rate by hour of day")
    ax.set_xticks(range(0, 24, 2))
    save(fig, FIG / "eda_ulb_fraud_by_hour.png")

    # Correlation of each feature with the label (Pearson, computed by Spark)
    feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    correlations = {c: df.stat.corr(c, LABEL) for c in feature_columns}
    top = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)[:12]
    results["top_correlations"] = {k: round(v, 4) for k, v in top}
    print(f"  Strongest correlations with fraud: {list(results['top_correlations'].items())[:5]}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [name for name, _ in top][::-1]
    values = [value for _, value in top][::-1]
    ax.barh(names, values, color=[FRAUD_COLOR if v > 0 else MAIN_COLOR for v in values], height=0.6)
    ax.axvline(0, color="#898781", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with fraud label")
    ax.set_title("ULB - Features most correlated with fraud")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "eda_ulb_feature_correlation.png")

    df.unpersist()
    return results


# --------------------------------------------------------------------------
# Dataset 2 - Sparkov
# --------------------------------------------------------------------------
def rate_chart(pdf, x, title, xlabel, path, rotate=False, horizontal=False):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if horizontal:
        ax.barh(pdf[x].astype(str), pdf["fraud_rate_pct"], color=MAIN_COLOR, height=0.6)
        ax.set_xlabel("Fraud rate (%)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    else:
        ax.bar(pdf[x].astype(str), pdf["fraud_rate_pct"], color=MAIN_COLOR, width=0.65)
        ax.set_ylabel("Fraud rate (%)")
        ax.set_xlabel(xlabel)
        if rotate:
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title(title)
    save(fig, path)


def grouped_rate(df, column):
    pdf = (
        df.groupBy(column)
        .agg(F.count("*").alias("transactions"), F.sum(LABEL).alias("fraud"), F.sum("amt").alias("amount"))
        .toPandas()
    )
    pdf["fraud_rate_pct"] = pdf["fraud"] / pdf["transactions"] * 100
    return pdf


def eda_sparkov(spark) -> dict:
    section("EDA - DATASET 2 (Sparkov)")
    df = (
        spark.read.parquet(config.spark_path(config.SPARKOV_CLEAN))
        .withColumn("hour", F.hour("trans_ts"))
        .withColumn("day_of_week", F.date_format("trans_ts", "E"))
        .withColumn("month", F.date_format("trans_ts", "yyyy-MM"))
        .withColumn("age", F.floor(F.months_between("trans_ts", "dob") / 12).cast("int"))
        .withColumn(
            "age_group",
            F.when(F.col("age") < 25, "18-24").when(F.col("age") < 35, "25-34").when(F.col("age") < 45, "35-44")
            .when(F.col("age") < 55, "45-54").when(F.col("age") < 65, "55-64").otherwise("65+"),
        )
        .cache()
    )
    results = {}

    counts = {row[LABEL]: row["count"] for row in df.groupBy(LABEL).count().collect()}
    results["class_counts"] = {"legitimate": counts[0], "fraud": counts[1]}
    print(f"  Legitimate {counts[0]:,} | Fraud {counts[1]:,} | ratio 1:{counts[0] // counts[1]}")
    class_distribution_chart(counts, "Sparkov - Class distribution", FIG / "eda_sparkov_class_distribution.png")

    amount_stats = (
        df.groupBy(LABEL)
        .agg(F.mean("amt").alias("mean"), F.expr("percentile_approx(amt, 0.5)").alias("median"), F.max("amt").alias("max"))
        .collect()
    )
    results["amount_by_class"] = {("fraud" if r[LABEL] else "legitimate"): {k: round(r[k], 2) for k in ["mean", "median", "max"]} for r in amount_stats}
    print(f"  Amount by class: {results['amount_by_class']}")
    results["amount_t_test"] = welch_t_test(df, "amt")
    print(f"  Welch t-test on amount: t={results['amount_t_test']['t_statistic']}, p={results['amount_t_test']['p_value']:.2e}")

    # Amount distribution (log10 buckets) for each class
    amount_hist = (
        df.withColumn("bucket", F.round(F.log10("amt") * 4) / 4)
        .groupBy("bucket", LABEL).count().toPandas()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        part = amount_hist[amount_hist[LABEL] == label].sort_values("bucket")
        ax.plot(10 ** part["bucket"], part["count"] / part["count"].sum() * 100, color=color, linewidth=2,
                marker="o", markersize=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("% of transactions in class")
    ax.set_title("Sparkov - Amount distribution: fraud vs legitimate")
    ax.legend()
    save(fig, FIG / "eda_sparkov_amount_distribution.png")

    category = grouped_rate(df, "category").sort_values("fraud_rate_pct", ascending=False)
    results["fraud_rate_by_category"] = dict(zip(category["category"], category["fraud_rate_pct"].round(3)))
    rate_chart(category, "category", "Sparkov - Fraud rate by merchant category", "", FIG / "eda_sparkov_fraud_by_category.png", horizontal=True)

    hourly = grouped_rate(df, "hour").sort_values("hour")
    results["fraud_rate_by_hour"] = dict(zip(hourly["hour"].astype(int).astype(str), hourly["fraud_rate_pct"].round(3)))
    rate_chart(hourly, "hour", "Sparkov - Fraud rate by hour of day", "Hour of day", FIG / "eda_sparkov_fraud_by_hour.png")

    ages = grouped_rate(df, "age_group").sort_values("age_group")
    results["fraud_rate_by_age_group"] = dict(zip(ages["age_group"], ages["fraud_rate_pct"].round(3)))
    rate_chart(ages, "age_group", "Sparkov - Fraud rate by customer age group", "Age group", FIG / "eda_sparkov_fraud_by_age.png")

    days = grouped_rate(df, "day_of_week")
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = days.set_index("day_of_week").loc[day_order].reset_index()
    results["fraud_rate_by_day"] = dict(zip(days["day_of_week"], days["fraud_rate_pct"].round(3)))
    rate_chart(days, "day_of_week", "Sparkov - Fraud rate by day of week", "Day", FIG / "eda_sparkov_fraud_by_day.png")

    gender = grouped_rate(df, "gender")
    results["fraud_rate_by_gender"] = dict(zip(gender["gender"], gender["fraud_rate_pct"].round(3)))

    states = grouped_rate(df, "state").sort_values("fraud", ascending=False).head(10)
    results["top_states_by_fraud_count"] = dict(zip(states["state"], states["fraud"].astype(int)))

    monthly = grouped_rate(df, "month").sort_values("month")
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(monthly["month"], monthly["fraud"], color=FRAUD_COLOR, width=0.65)
    ax1.set_ylabel("Fraud transactions")
    ax1.set_title("Sparkov - Fraud transactions per month")
    plt.setp(ax1.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    save(fig, FIG / "eda_sparkov_fraud_by_month.png")
    results["fraud_by_month"] = dict(zip(monthly["month"], monthly["fraud"].astype(int)))

    # Chi-square test: is merchant category independent of fraud? (Spark MLlib ChiSquareTest)
    indexed = StringIndexer(inputCol="category", outputCol="category_idx").fit(df).transform(df)
    vector = VectorAssembler(inputCols=["category_idx"], outputCol="cat_vec").transform(indexed)
    chi = ChiSquareTest.test(vector, "cat_vec", LABEL).first()
    results["chi_square_category_vs_fraud"] = {
        "statistic": round(float(chi.statistics[0]), 2),
        "degrees_of_freedom": int(chi.degreesOfFreedom[0]),
        "p_value": float(chi.pValues[0]),
    }
    print(f"  Chi-square (category vs fraud): {results['chi_square_category_vs_fraud']}")

    # Key insights, derived from the numbers above
    top_category = category.iloc[0]
    worst_hours = hourly.sort_values("fraud_rate_pct", ascending=False).head(4)["hour"].astype(int).tolist()
    results["insights"] = [
        f"'{top_category['category']}' has the highest fraud rate ({top_category['fraud_rate_pct']:.2f}%).",
        f"Fraud concentrates at hours {sorted(worst_hours)} (late night).",
        f"Average fraud amount {results['amount_by_class']['fraud']['mean']} vs "
        f"{results['amount_by_class']['legitimate']['mean']} for legitimate transactions.",
    ]
    for insight in results["insights"]:
        print(f"  * {insight}")

    df.unpersist()
    return results


def main() -> None:
    start = time.time()
    config.ensure_dirs()
    spark = get_spark("CreditGuard-EDA")
    stats = {"ulb": eda_ulb(spark), "sparkov": eda_sparkov(spark)}
    (config.RESULTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    spark.stop()
    print(f"\nEDA finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
