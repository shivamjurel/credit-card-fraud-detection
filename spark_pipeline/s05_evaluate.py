"""
Step 5 - Model Evaluation Charts
================================

Run:  python spark_pipeline/s05_evaluate.py   (after s04_train_models.py)

Reads the metrics and test-set scores saved by step 4 and draws, for each dataset:
  * model_comparison_<ds>.png     grouped bar chart: Precision / Recall / F1 / ROC-AUC / PR-AUC
  * confusion_matrices_<ds>.png   one confusion matrix per model
  * roc_curves_<ds>.png           ROC curves of all models on one chart
  * pr_curves_<ds>.png            Precision-Recall curves (more informative for rare fraud)
  * feature_importance_<ds>.png   what the selected model relies on most
  * training_time.png             training time per model (scalability discussion)
"""

import json

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import config
from plot_style import FRAUD_COLOR, MAIN_COLOR, MODEL_COLORS, plt, save

FIG = config.FIGURES_DIR
SCORES_DIR = config.RESULTS_DIR / "scores"
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1-score"),
           ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]


def score_file(dataset: str, model_name: str):
    return SCORES_DIR / f"{dataset}_{model_name.lower().replace(' ', '_').replace('-', '_')}.npz"


def model_comparison_chart(dataset: str, data: dict) -> None:
    models = list(data["models"])
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, model_name in enumerate(models):
        values = [data["models"][model_name][key] for key, _ in METRICS]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width * 0.92,
                      color=MODEL_COLORS[i], label=model_name + (" ★" if model_name == data["selected_model"] else ""))
        if model_name == data["selected_model"]:
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), ha="center",
                            va="bottom", fontsize=7)
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (test set)")
    ax.set_title(f"{config.DATASETS[dataset]['name']} - model comparison (★ = selected)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, FIG / f"model_comparison_{dataset}.png")


def confusion_matrix_grid(dataset: str, data: dict) -> None:
    models = list(data["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.3))
    for ax, model_name in zip(axes, models):
        cm = data["models"][model_name]["confusion_matrix"]
        matrix = np.array([[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]])
        row_share = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, f"{value:,}\n({row_share[r, c]:.1%})", ha="center", va="center", fontsize=8,
                    color="white" if row_share[r, c] > 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_title(model_name + (" ★" if model_name == data["selected_model"] else ""), fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel("Actual")
    fig.suptitle(f"{config.DATASETS[dataset]['name']} - confusion matrices (test set)", fontweight="bold")
    fig.tight_layout()
    save(fig, FIG / f"confusion_matrices_{dataset}.png")


def curve_charts(dataset: str, data: dict) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(5.8, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(5.8, 5))
    for i, model_name in enumerate(data["models"]):
        path = score_file(dataset, model_name)
        if not path.exists():
            continue
        saved = np.load(path)
        scores, labels = saved["score"], saved["label"]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[i], linewidth=2, label=f"{model_name} (AUC {auc(fpr, tpr):.3f})")
        ax_pr.plot(recall, precision, color=MODEL_COLORS[i], linewidth=2,
                   label=f"{model_name} (AUC {auc(recall, precision):.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#898781", linewidth=1, label="Random guess")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)",
               title=f"ROC curves - {dataset.upper()}")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(axis="x")
    save(fig_roc, FIG / f"roc_curves_{dataset}.png")

    ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall curves - {dataset.upper()}",
              ylim=(0, 1.02))
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(axis="x")
    save(fig_pr, FIG / f"pr_curves_{dataset}.png")


def feature_importance_chart(dataset: str, data: dict) -> None:
    importances = data.get("feature_importances") or {}
    if not importances:
        return
    names = list(importances)[:12][::-1]
    values = [importances[name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(names, values, color=MAIN_COLOR, height=0.6)
    ax.set_xlabel("Importance")
    ax.set_title(f"{data['selected_model']} - top features ({dataset.upper()})")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, FIG / f"feature_importance_{dataset}.png")


def training_time_chart(results: dict) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (dataset, data) in zip(axes, results.items()):
        models = list(data["models"])
        seconds = [data["models"][m]["train_seconds"] for m in models]
        ax.barh(models, seconds, color=[MODEL_COLORS[i] for i in range(len(models))], height=0.6)
        ax.set_xlabel("Training time (seconds, local Spark)")
        ax.set_title(dataset.upper())
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, FIG / "training_time.png")


def main() -> None:
    results = json.loads((config.RESULTS_DIR / "model_comparison.json").read_text())
    for dataset, data in results.items():
        print(f"\n{dataset.upper()} - selected: {data['selected_model']}")
        model_comparison_chart(dataset, data)
        confusion_matrix_grid(dataset, data)
        curve_charts(dataset, data)
        feature_importance_chart(dataset, data)
    training_time_chart(results)


if __name__ == "__main__":
    main()
