"""
KernelFlow — Benchmark Dashboard Visualization
=================================================
Reads benchmark/results.csv (produced by bench_inference) and generates a
4‑panel dashboard saved as benchmark_dashboard.png.

CSV columns: benchmark, variant, parameter, time_ms, metric_value, metric_unit

Panels:
  1. GEMM comparison — grouped bar chart (Naive vs Tiled vs cuBLAS) at each size
  2. Forward‑pass latency — grouped bars (CPU / GPU / Streams / Graphs × batch)
  3. RL training speedup — horizontal bar CPU vs GPU
  4. Optimisation waterfall — cumulative latency reduction from each technique

Usage:
    python benchmark/plot_results.py                       # default path
    python benchmark/plot_results.py --input results.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Professional dark‑theme palette ──────────────────────────────────────────
BG_DARK = "#1a1a2e"
BG_AXES = "#16213e"
GPU_GREEN = "#76b900"
GPU_GREEN_LIGHT = "#a3d94e"
TILED_BLUE = "#00a8e8"
CUBLAS_GOLD = "#ffd700"
CPU_GRAY = "#8e8e8e"
STREAM_PURPLE = "#bb86fc"
GRAPH_ORANGE = "#ff8c00"
RL_CPU_RED = "#ff4c4c"
RL_GPU_GREEN = "#76b900"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
WATERFALL_SAVE = "#2ecc71"
WATERFALL_REMAIN = "#3498db"

# Variant → colour mapping
GEMM_COLORS = {
    "Naive_GPU": CPU_GRAY,
    "Tiled_GPU": TILED_BLUE,
    "cuBLAS": CUBLAS_GOLD,
}
FWD_COLORS = {
    "CPU": CPU_GRAY,
    "GPU_Basic": GPU_GREEN,
    "GPU_Stream": STREAM_PURPLE,
    "GPU_Graph": GRAPH_ORANGE,
}
FWD_LABELS = {
    "CPU": "CPU",
    "GPU_Basic": "GPU Basic",
    "GPU_Stream": "GPU + Streams",
    "GPU_Graph": "GPU + Graphs",
}
GEMM_LABELS = {
    "Naive_GPU": "Naive GPU",
    "Tiled_GPU": "Tiled (shared mem)",
    "cuBLAS": "cuBLAS",
}


def _apply_style() -> None:
    """Set global matplotlib dark theme."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.35,
        "font.family": "monospace",
        "font.size": 10,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 1 — GEMM Performance (GFLOPS bar chart)
# ═════════════════════════════════════════════════════════════════════════════
def panel_gemm(ax: plt.Axes, df: pd.DataFrame) -> None:
    gemm = df[df["benchmark"] == "GEMM"].copy()
    # Keep GPU variants only (CPU is baseline reference, very slow)
    gemm = gemm[gemm["variant"].isin(GEMM_COLORS.keys())]
    if gemm.empty:
        ax.text(0.5, 0.5, "No GEMM data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    sizes = sorted(gemm["parameter"].unique(), key=lambda x: int(x))
    variants = [v for v in GEMM_COLORS if v in gemm["variant"].values]
    n_var = len(variants)
    x = np.arange(len(sizes))
    width = 0.7 / max(n_var, 1)

    for i, var in enumerate(variants):
        vals = []
        for sz in sizes:
            row = gemm[(gemm["variant"] == var) & (gemm["parameter"] == sz)]
            vals.append(row["metric_value"].values[0] if len(row) else 0)
        offset = (i - (n_var - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=GEMM_LABELS.get(var, var),
                      color=GEMM_COLORS[var], edgecolor="white", linewidth=0.4, alpha=0.9)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM Performance — Naive vs Tiled vs cuBLAS", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 2 — Forward Pass Latency (grouped bars)
# ═════════════════════════════════════════════════════════════════════════════
def panel_forward(ax: plt.Axes, df: pd.DataFrame) -> None:
    fwd = df[df["benchmark"] == "Forward"].copy()
    if fwd.empty:
        ax.text(0.5, 0.5, "No Forward data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    batches = sorted(fwd["parameter"].unique(), key=lambda x: int(x))
    variants = [v for v in FWD_COLORS if v in fwd["variant"].values]
    n_var = len(variants)
    x = np.arange(len(batches))
    width = 0.7 / max(n_var, 1)

    for i, var in enumerate(variants):
        vals = []
        for bs in batches:
            row = fwd[(fwd["variant"] == var) & (fwd["parameter"] == bs)]
            vals.append(row["time_ms"].values[0] if len(row) else 0)
        offset = (i - (n_var - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=FWD_LABELS.get(var, var),
                      color=FWD_COLORS[var], edgecolor="white", linewidth=0.4, alpha=0.9)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=TEXT_COLOR,
                        rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in batches])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Forward Pass Latency — 4 Methods × 4 Batch Sizes", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 3 — RL Training Speedup (horizontal bars)
# ═════════════════════════════════════════════════════════════════════════════
def panel_rl(ax: plt.Axes, df: pd.DataFrame) -> None:
    rl = df[(df["benchmark"] == "RL") & (df["parameter"] == "steps_per_sec")].copy()
    if rl.empty:
        ax.text(0.5, 0.5, "No RL data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    labels = []
    values = []
    colors = []
    color_map = {"CPU_1env": RL_CPU_RED, "GPU_512env": RL_GPU_GREEN}

    for _, row in rl.iterrows():
        var = row["variant"]
        labels.append(var.replace("_", " "))
        values.append(row["metric_value"])
        colors.append(color_map.get(var, CPU_GRAY))

    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=0.5, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f} steps/s", va="center", fontsize=10, color=TEXT_COLOR,
                fontweight="bold")

    # Speedup annotation
    if len(values) == 2 and values[0] > 0:
        speedup = values[1] / values[0]
        ax.text(0.95, 0.85, f"{speedup:.0f}× speedup", transform=ax.transAxes,
                ha="right", va="top", fontsize=14, fontweight="bold", color=GPU_GREEN,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_AXES,
                          edgecolor=GPU_GREEN, alpha=0.9))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Environment Steps / Second")
    ax.set_title("RL Training — CPU Sequential vs GPU Parallel", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(left=0)


# ═════════════════════════════════════════════════════════════════════════════
#  PANEL 4 — Optimisation Waterfall Chart
# ═════════════════════════════════════════════════════════════════════════════
def panel_waterfall(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Show how each optimisation stage reduces forward‑pass latency.

    Uses the largest batch size available for the most dramatic comparison.
    """
    fwd = df[df["benchmark"] == "Forward"].copy()
    if fwd.empty:
        ax.text(0.5, 0.5, "No Forward data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    # Pick the largest batch
    max_batch = sorted(fwd["parameter"].unique(), key=lambda x: int(x))[-1]
    fwd = fwd[fwd["parameter"] == max_batch]

    order = ["CPU", "GPU_Basic", "GPU_Stream", "GPU_Graph"]
    stage_labels = ["CPU Baseline", "→ GPU Forward", "→ + CUDA Streams", "→ + CUDA Graphs"]
    times = {}
    for var in order:
        row = fwd[fwd["variant"] == var]
        if len(row):
            times[var] = row["time_ms"].values[0]

    if "CPU" not in times:
        ax.text(0.5, 0.5, "Incomplete Forward data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    # Build waterfall: baseline, then deltas (savings)
    bars_bottom = []
    bars_height = []
    bar_colors = []
    bar_labels_text = []
    prev = times["CPU"]
    for i, var in enumerate(order):
        t = times.get(var, prev)
        if i == 0:
            bars_bottom.append(0)
            bars_height.append(t)
            bar_colors.append(CPU_GRAY)
            bar_labels_text.append(f"{t:.3f} ms")
        else:
            saving = prev - t
            bars_bottom.append(t)
            bars_height.append(saving if saving > 0 else 0)
            bar_colors.append(WATERFALL_SAVE)
            if saving > 0:
                bar_labels_text.append(f"−{saving:.3f} ms")
            else:
                bar_labels_text.append(f"{t:.3f} ms")
            prev = t

    # Add final remaining bar
    final = times.get(order[-1], prev)

    x = np.arange(len(order))
    bars = ax.bar(x, bars_height, bottom=bars_bottom, width=0.55, color=bar_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.85)

    # Remaining time (final latency) overlay on last bar
    ax.bar(len(order) - 1, final, bottom=0, width=0.55, color=WATERFALL_REMAIN,
           edgecolor="white", linewidth=0.5, alpha=0.6, zorder=1)

    # Connector lines between bars
    for i in range(len(order) - 1):
        y_connect = bars_bottom[i + 1] + bars_height[i + 1]
        ax.plot([i + 0.3, i + 0.7], [y_connect, y_connect],
                color=TEXT_COLOR, linewidth=0.8, linestyle=":", alpha=0.5)

    # Labels
    for i, (bar, txt) in enumerate(zip(bars, bar_labels_text)):
        y_pos = bars_bottom[i] + bars_height[i] + times["CPU"] * 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, txt,
                ha="center", va="bottom", fontsize=9, color=TEXT_COLOR, fontweight="bold")

    # Total speedup annotation
    total_speedup = times["CPU"] / final if final > 0 else 0
    ax.text(0.95, 0.92, f"Total: {total_speedup:.1f}× faster\n(batch={max_batch})",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold", color=GPU_GREEN,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_AXES,
                      edgecolor=GPU_GREEN, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Optimisation Waterfall — Cumulative Latency Reduction", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KernelFlow benchmark dashboard")
    parser.add_argument("--input", default="benchmark/results.csv",
                        help="Path to results.csv")
    parser.add_argument("--output", default="benchmark_dashboard.png",
                        help="Output image path")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run bench_inference first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"benchmark", "variant", "parameter", "time_ms", "metric_value", "metric_unit"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns {missing} in {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure parameter is string for grouping
    df["parameter"] = df["parameter"].astype(str)

    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("KernelFlow — Benchmark Dashboard", fontsize=18,
                 fontweight="bold", color=GPU_GREEN, y=0.97)

    panel_gemm(axes[0, 0], df)
    panel_forward(axes[0, 1], df)
    panel_rl(axes[1, 0], df)
    panel_waterfall(axes[1, 1], df)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
