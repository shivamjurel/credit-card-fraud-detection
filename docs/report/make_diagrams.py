"""
Diagrams for the CreditGuard report (Chapter 2).
Run:  python docs/report/make_diagrams.py
Output: docs/report/figures/fig_architecture.png, fig_workflow.png, fig_usecase.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
BLUE, BLUE_L = "#1c5cab", "#e6f0fc"
GREEN, GREEN_L = "#067a06", "#e7f6e7"
ORANGE, ORANGE_L = "#b25a1a", "#fdf0e6"
GREY, GREY_L = "#52514e", "#f2f2ef"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})


def box(ax, x, y, w, h, text, edge=BLUE, fill=BLUE_L, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    lines = text.split("\n")
    for i, line in enumerate(lines):
        offset = (len(lines) - 1) / 2 - i
        ax.text(x + w / 2, y + h / 2 + offset * 0.22, line, ha="center", va="center", fontsize=size,
                fontweight="bold" if i == 0 else "normal", color="#111")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=GREY))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
    for y, name in [(6.0, "Data\nSources"), (4.6, "Spark batch\npipeline"), (3.05, "Model\nlayer"), (1.2, "Serving &\nBI")]:
        ax.text(0.05, y + 0.35, name, fontsize=9, color=GREY, fontweight="bold", va="center")
    box(ax, 1.5, 5.9, 3.9, 0.85, "Dataset 1: ULB (Kaggle)\n284,807 real transactions, PCA features", GREY, GREY_L, 9)
    box(ax, 5.9, 5.9, 4.5, 0.85, "Dataset 2: Sparkov (Kaggle)\n1,852,394 transactions, readable features", GREY, GREY_L, 9)
    box(ax, 1.5, 4.45, 2.8, 0.95, "Ingestion & cleaning\nschema, duplicates,\nPII removal -> Parquet", BLUE, BLUE_L, 8.5)
    box(ax, 4.55, 4.45, 2.6, 0.95, "EDA & statistics\nSpark SQL, t-test,\nchi-square", BLUE, BLUE_L, 8.5)
    box(ax, 7.4, 4.45, 3.0, 0.95, "Feature engineering\nwindow functions:\ncard history, time, distance", BLUE, BLUE_L, 8.5)
    box(ax, 1.5, 2.85, 5.6, 1.05, "Spark MLlib: 5 models\nLogistic Regression, Decision Tree, Random Forest,\nGradient-Boosted Trees, Linear SVM", BLUE, BLUE_L, 8.8)
    box(ax, 7.4, 2.85, 3.0, 1.05, "Evaluation & selection\nclass weights, threshold\ntuning, best F1", BLUE, BLUE_L, 8.5)
    box(ax, 1.5, 0.9, 2.8, 1.1, "Web application\nFastAPI + React\nsingle-transaction check", GREEN, GREEN_L, 8.5)
    box(ax, 4.55, 0.9, 2.6, 1.1, "Structured Streaming\nlive scoring +\nfraud alerts", GREEN, GREEN_L, 8.5)
    box(ax, 7.4, 0.9, 3.0, 1.1, "Power BI dashboard\nKPIs, trends, maps,\nmodel performance", GREEN, GREEN_L, 8.5)
    ax.text(9.05, 2.42, "saved best\nPipelineModel", fontsize=8, color=GREY, ha="left", va="center")
    arrow(ax, 3.4, 5.9, 2.9, 5.4); arrow(ax, 8.2, 5.9, 3.6, 5.4)
    arrow(ax, 4.3, 4.92, 4.55, 4.92); arrow(ax, 7.15, 4.92, 7.4, 4.92)
    arrow(ax, 8.9, 4.45, 5.6, 3.9); arrow(ax, 7.1, 3.37, 7.4, 3.37)
    arrow(ax, 8.9, 2.85, 2.9, 2.0); arrow(ax, 8.9, 2.85, 5.85, 2.0); arrow(ax, 8.9, 2.85, 8.9, 2.0)
    ax.text(5.95, 0.35, "Predictions from the web app and the stream are stored in SQLite and shown on the live dashboard",
            fontsize=8, color=GREY, ha="center", style="italic")
    fig.savefig(OUT / "fig_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def workflow():
    steps = [
        ("Data collection", "2 Kaggle datasets,\n2.1M transactions"),
        ("Pre-processing", "schema, duplicates,\nPII removal, Parquet"),
        ("EDA & statistics", "imbalance, amount, hour,\ncategory, tests"),
        ("Feature engineering", "window functions,\ntime, age, distance"),
        ("Model training", "5 Spark MLlib models\n+ class weights"),
        ("Evaluation", "threshold on validation,\nP / R / F1 / PR-AUC"),
        ("Deployment", "web app, streaming,\nPower BI"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11); ax.set_ylim(-0.55, 4.8); ax.axis("off")
    w, h, gap = 2.3, 1.2, 0.45
    positions = [(0.2 + i * (w + gap), 3.3) for i in range(4)] + [(0.2 + (3 - i) * (w + gap), 0.6) for i in range(3)]
    for i, ((title, sub), (x, y)) in enumerate(zip(steps, positions)):
        colour = (GREEN, GREEN_L) if i == len(steps) - 1 else (BLUE, BLUE_L)
        box(ax, x, y, w, h, f"{i + 1}. {title}\n{sub}", *colour, size=9)
    for i in range(3):
        x, y = positions[i]
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    x, y = positions[3]
    arrow(ax, x + w / 2, y, x + w / 2, 0.6 + h)
    for i in range(4, 6):
        x, y = positions[i]
        arrow(ax, x, y + h / 2, x - gap, y + h / 2)
    ex, ey = positions[5]; mx, my = positions[4]
    ax.add_patch(FancyArrowPatch((ex + w / 2, ey), (mx + w / 2, my), connectionstyle="arc3,rad=0.45",
                                 arrowstyle="-|>", mutation_scale=12, color=ORANGE, linewidth=1.2, linestyle="--"))
    ax.text((ex + mx + w) / 2, -0.45, "tune features / parameters if validation F1 is low", fontsize=8.5,
            color=ORANGE, ha="center")
    fig.savefig(OUT / "fig_workflow.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def use_case():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.6, 0.3), 4.8, 5.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=1.3, edgecolor=GREY, facecolor="white"))
    ax.text(5.0, 5.05, "CreditGuard System", ha="center", fontsize=10, fontweight="bold")

    def actor(x, y, name):
        ax.add_patch(plt.Circle((x, y + 0.55), 0.16, fill=False, linewidth=1.3))
        ax.plot([x, x], [y + 0.39, y - 0.05], color="#111", linewidth=1.3)
        ax.plot([x - 0.25, x + 0.25], [y + 0.25, y + 0.25], color="#111", linewidth=1.3)
        ax.plot([x, x - 0.2], [y - 0.05, y - 0.35], color="#111", linewidth=1.3)
        ax.plot([x, x + 0.2], [y - 0.05, y - 0.35], color="#111", linewidth=1.3)
        ax.text(x, y - 0.45, name, ha="center", va="top", fontsize=9)

    cases = [(5.0, 4.45, "Check a transaction (score + risk)"), (5.0, 3.7, "Receive real-time fraud alerts"),
             (5.0, 2.95, "View prediction history & dashboard"), (5.0, 2.2, "Analyse fraud trends in Power BI"),
             (5.0, 1.45, "Run pipeline & retrain models"), (5.0, 0.75, "Compare 5 models (metrics, charts)")]
    for x, y, text in cases:
        ax.add_patch(Ellipse((x, y), 4.0, 0.55, facecolor=BLUE_L, edgecolor=BLUE, linewidth=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.8)
    actor(1.2, 3.6, "Fraud analyst")
    actor(1.2, 1.4, "Bank manager")
    actor(8.8, 1.6, "Data engineer")
    for idx in (0, 1, 2):
        ax.plot([1.45, cases[idx][0] - 2.0], [3.85, cases[idx][1]], color=GREY, linewidth=1)
    for idx in (2, 3):
        ax.plot([1.45, cases[idx][0] - 2.0], [1.65, cases[idx][1]], color=GREY, linewidth=1)
    for idx in (4, 5):
        ax.plot([8.55, cases[idx][0] + 2.0], [1.85, cases[idx][1]], color=GREY, linewidth=1)
    fig.savefig(OUT / "fig_usecase.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture(); workflow(); use_case()
    print("diagrams ->", OUT)
