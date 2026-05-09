"""Generate publication-quality plots from benchmark results.

Creates 4 figures:
1. Bar chart: ROUGE-L F1 and Faithfulness by strategy
2. Scatter: Faithfulness vs Latency tradeoff
3. Line chart: Guardrail trigger rate vs threshold (from sweep)
4. Box plot: Confidence score distribution by strategy

Usage:
    python experiments/generate_plots.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

RESULTS_DIR = Path("./results")
EXPERIMENTS_DIR = Path("./results/experiments")
OUTPUT_DIR = Path("./docs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]


def plot_generation_metrics():
    """Bar chart: ROUGE-L F1 and Faithfulness by strategy."""
    agg = pd.read_csv(RESULTS_DIR / "aggregated_metrics.csv", index_col=0)

    strategies = list(agg.index)
    rouge = []
    faith = []
    for s in strategies:
        r = agg.loc[s].get("rouge_l_f1_mean", 0)
        f = agg.loc[s].get("faithfulness_mean", 0)
        rouge.append(r if pd.notna(r) else 0)
        faith.append(f if pd.notna(f) else 0)

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, rouge, width, label="ROUGE-L F1", color=COLORS[0])
    bars2 = ax.bar(x + width / 2, faith, width, label="Faithfulness", color=COLORS[1])

    ax.set_ylabel("Score")
    ax.set_title("Generation Quality by RAG Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(rouge), max(faith)) * 1.3 + 0.05)

    # Value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    path = OUTPUT_DIR / "generation_metrics.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_latency_vs_faithfulness():
    """Scatter: Faithfulness vs Latency tradeoff."""
    agg = pd.read_csv(RESULTS_DIR / "aggregated_metrics.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strategy in enumerate(agg.index):
        lat = agg.loc[strategy].get("latency_ms_mean", 0)
        faith = agg.loc[strategy].get("faithfulness_mean", 0)
        conf = agg.loc[strategy].get("confidence_score_mean", 0)
        trigger = agg.loc[strategy].get("guardrail_trigger_rate", 1)

        if pd.isna(faith) or faith == 0:
            faith = 0
            marker = "x"
        else:
            marker = "o"

        size = max(100, (1 - trigger) * 500)
        ax.scatter(lat, faith, s=size, c=COLORS[i], marker=marker,
                   label=f"{strategy} (trigger={trigger:.0%})", zorder=5)
        ax.annotate(strategy, (lat, faith), fontsize=9,
                    xytext=(8, 8), textcoords="offset points")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Faithfulness Score")
    ax.set_title("Faithfulness vs Latency Tradeoff\n(Bubble size = proportion of answered queries)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "faithfulness_vs_latency.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_threshold_sweep():
    """Line chart: Guardrail trigger rate vs threshold."""
    sweep_path = EXPERIMENTS_DIR / "threshold_sweep.csv"
    if not sweep_path.exists():
        print(f"Skipping threshold sweep plot - run experiments/threshold_sweep.py first")
        return

    df = pd.read_csv(sweep_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: trigger rate
    for i, strategy in enumerate(df["strategy"].unique()):
        subset = df[df["strategy"] == strategy]
        ax1.plot(subset["threshold"], subset["guardrail_trigger_rate"],
                 marker="o", color=COLORS[i % len(COLORS)], label=strategy, linewidth=2)

    ax1.set_xlabel("Guardrail Threshold")
    ax1.set_ylabel("Guardrail Trigger Rate")
    ax1.set_title("Guardrail Trigger Rate vs Threshold")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_xlim(0.05, 0.85)
    ax1.grid(True, alpha=0.3)

    # Right: ROUGE-L F1
    for i, strategy in enumerate(df["strategy"].unique()):
        subset = df[df["strategy"] == strategy].dropna(subset=["rouge_l_f1"])
        if not subset.empty:
            ax2.plot(subset["threshold"], subset["rouge_l_f1"],
                     marker="s", color=COLORS[i % len(COLORS)], label=strategy, linewidth=2)

    ax2.set_xlabel("Guardrail Threshold")
    ax2.set_ylabel("ROUGE-L F1 (answered queries only)")
    ax2.set_title("Answer Quality vs Threshold")
    ax2.legend()
    ax2.set_xlim(0.05, 0.85)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "threshold_sweep.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_confidence_distribution():
    """Per-strategy confidence score distributions from raw results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for csv_name in ["Baseline_results.csv", "Hybrid_results.csv",
                     "Reranker_results.csv", "Query_Decomposition_results.csv"]:
        path = RESULTS_DIR / csv_name
        if path.exists():
            df = pd.read_csv(path)
            scores = df["confidence_score"].dropna().values
            if len(scores) > 0:
                data.append(scores)
                labels.append(csv_name.replace("_results.csv", ""))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)

        ax.axhline(y=0.6, color="red", linestyle="--", alpha=0.7,
                   label="Default threshold (0.6)")
        ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7,
                   label="Tuned threshold (0.3)")
        ax.set_ylabel("Confidence Score (max retrieval similarity)")
        ax.set_title("Confidence Score Distribution by Strategy\nwith Guardrail Thresholds")
        ax.legend()

    plt.tight_layout()
    path = OUTPUT_DIR / "confidence_distribution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating plots from benchmark results...\n")
    plot_generation_metrics()
    plot_latency_vs_faithfulness()
    plot_threshold_sweep()
    plot_confidence_distribution()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
