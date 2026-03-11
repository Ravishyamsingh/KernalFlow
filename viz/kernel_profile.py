"""
KernelFlow -- Kernel Profile Visualization
===========================================
Reads benchmark/results.csv (produced by bench_inference.exe) and generates
kernel-level performance charts.

CSV columns: benchmark, variant, parameter, time_ms, metric_value, metric_unit

Panels:
  1. GEMM GFLOPS scaling — line chart across matrix sizes
  2. GEMM latency — grouped bar chart (log scale)
  3. Activation kernel speedup — CPU vs GPU
  4. Tiled GEMM efficiency vs cuBLAS — percentage bars

Usage:
    python viz/kernel_profile.py
    python viz/kernel_profile.py --input benchmark/results.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ?? Professional dark-theme palette ??????????????????????????????????????????
BG_DARK = "#1a1a2e"
BG_AXES = "#16213e"
GPU_GREEN = "#76b900"
TILED_BLUE = "#00a8e8"
CUBLAS_GOLD = "#ffd700"
CPU_GRAY = "#8e8e8e"
NAIVE_RED = "#ff6b6b"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"


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


# ?????????????????????????????????????????????????????????????????????????????
#  PANEL 1 — GEMM GFLOPS Scaling (line chart)
# ?????????????????????????????????????????????????????????????????????????????
def panel_gemm_scaling(ax: plt.Axes, df: pd.DataFrame) -> None:
    gemm = df[df["benchmark"] == "GEMM"].copy()
    if gemm.empty:
        ax.text(0.5, 0.5, "No GEMM data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    # Plot each GPU variant found in the data
    variant_style = {
        "Naive_GPU": (NAIVE_RED, "s", "Naive GPU"),
        "Tiled_GPU": (TILED_BLUE, "D", "Tiled (shared mem)"),
        "cuBLAS":    (CUBLAS_GOLD, "o", "cuBLAS"),
    }
    for var, (color, marker, label) in variant_style.items():
        data = gemm[gemm["variant"] == var].sort_values(
            "parameter", key=lambda s: s.astype(int))
        if data.empty:
            continue
        sizes = data["parameter"].astype(int).values
        gflops = data["metric_value"].values
        ax.plot(sizes, gflops, color=color, marker=marker, markersize=8,
                linewidth=2, label=label, zorder=3)
        for s, g in zip(sizes, gflops):
            ax.annotate(f"{g:.0f}", (s, g), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=color)

    ax.set_xlabel("Matrix Size (N x N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM Kernel Scaling", fontweight="bold", fontsize=11,
                 color=GPU_GREEN)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=GRID_COLOR)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)


# ?????????????????????????????????????????????????????????????????????????????
#  PANEL 2 — GEMM Latency (log-scale grouped bars)
# ?????????????????????????????????????????????????????????????????????????????
def panel_gemm_latency(ax: plt.Axes, df: pd.DataFrame) -> None:
    gemm = df[df["benchmark"] == "GEMM"].copy()
    if gemm.empty:
        ax.text(0.5, 0.5, "No GEMM data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    variant_style = {
        "CPU":       (CPU_GRAY,    "CPU"),
        "Naive_GPU": (NAIVE_RED,   "Naive GPU"),
        "Tiled_GPU": (TILED_BLUE,  "Tiled GPU"),
        "cuBLAS":    (CUBLAS_GOLD, "cuBLAS"),
    }

    sizes = sorted(gemm["parameter"].unique(), key=lambda x: int(x))
    variants = [v for v in variant_style if v in gemm["variant"].values]
    n_var = len(variants)
    x = np.arange(len(sizes))
    width = 0.7 / max(n_var, 1)

    for i, var in enumerate(variants):
        vals = []
        for sz in sizes:
            row = gemm[(gemm["variant"] == var) & (gemm["parameter"] == sz)]
            vals.append(row["time_ms"].values[0] if len(row) else np.nan)
        offset = (i - (n_var - 1) / 2) * width
        color, label = variant_style[var]
        ax.bar(x + offset, vals, width * 0.9, label=label,
               color=color, edgecolor="white", linewidth=0.4, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms) - log scale")
    ax.set_yscale("log")
    ax.set_title("GEMM Latency (Log Scale)", fontweight="bold", fontsize=11,
                 color=GPU_GREEN)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)


# ?????????????????????????????????????????????????????????????????????????????
#  PANEL 3 — Activation Kernels (CPU vs GPU)
# ?????????????????????????????????????????????????????????????????????????????
def panel_activations(ax: plt.Axes, df: pd.DataFrame) -> None:
    act = df[df["benchmark"] == "Activation"].copy()
    if act.empty:
        ax.text(0.5, 0.5, "No Activation data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    # Discover kernel names dynamically from the data
    cpu_variants = [v for v in act["variant"].unique() if v.startswith("CPU_")]
    kernels = [v.replace("CPU_", "") for v in cpu_variants]

    cpu_times = []
    gpu_times = []
    for k in kernels:
        cpu_row = act[act["variant"] == f"CPU_{k}"]
        gpu_row = act[act["variant"] == f"GPU_{k}"]
        cpu_times.append(cpu_row["time_ms"].values[0] if len(cpu_row) else 0)
        gpu_times.append(gpu_row["time_ms"].values[0] if len(gpu_row) else 0)

    x = np.arange(len(kernels))
    width = 0.3

    ax.bar(x - width / 2, cpu_times, width, label="CPU",
           color=CPU_GRAY, edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, gpu_times, width, label="GPU",
           color=GPU_GREEN, edgecolor="white", linewidth=0.4)

    # Speedup annotations
    for i, (ct, gt) in enumerate(zip(cpu_times, gpu_times)):
        if gt > 0:
            speedup = ct / gt
            ax.text(i + width / 2, gt + max(cpu_times) * 0.03,
                    f"{speedup:.0f}x", ha="center", va="bottom",
                    fontsize=11, color=GPU_GREEN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Activation Kernels - CPU vs GPU", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.legend(framealpha=0.3, edgecolor=GRID_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)


# ?????????????????????????????????????????????????????????????????????????????
#  PANEL 4 — Tiled GEMM Efficiency vs cuBLAS
# ?????????????????????????????????????????????????????????????????????????????
def panel_efficiency(ax: plt.Axes, df: pd.DataFrame) -> None:
    gemm = df[df["benchmark"] == "GEMM"].copy()
    if gemm.empty:
        ax.text(0.5, 0.5, "No GEMM data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    sizes = sorted(gemm["parameter"].unique(), key=lambda x: int(x))
    efficiency = []
    valid_sizes = []

    for sz in sizes:
        tiled = gemm[(gemm["variant"] == "Tiled_GPU") & (gemm["parameter"] == sz)]
        cublas = gemm[(gemm["variant"] == "cuBLAS") & (gemm["parameter"] == sz)]
        if len(tiled) and len(cublas) and cublas["metric_value"].values[0] > 0:
            eff = (tiled["metric_value"].values[0] /
                   cublas["metric_value"].values[0]) * 100
            efficiency.append(eff)
            valid_sizes.append(int(sz))

    if not valid_sizes:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR, fontsize=14)
        return

    colors = [TILED_BLUE if e < 50 else GPU_GREEN for e in efficiency]
    bars = ax.bar(range(len(valid_sizes)), efficiency, color=colors,
                  edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, eff in zip(bars, efficiency):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{eff:.1f}%", ha="center", va="bottom", fontsize=10,
                color=TEXT_COLOR, fontweight="bold")

    ax.axhline(y=100, color=CUBLAS_GOLD, linestyle="--", linewidth=1,
               alpha=0.7, label="cuBLAS (100%)")
    ax.set_xticks(range(len(valid_sizes)))
    ax.set_xticklabels([f"{s}x{s}" for s in valid_sizes])
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Efficiency vs cuBLAS (%)")
    ax.set_title("Tiled GEMM Efficiency vs cuBLAS", fontweight="bold",
                 fontsize=11, color=GPU_GREEN)
    ax.legend(framealpha=0.3, edgecolor=GRID_COLOR)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 110)


# ?????????????????????????????????????????????????????????????????????????????
#  MAIN
# ?????????????????????????????????????????????????????????????????????????????
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KernelFlow kernel profiles")
    parser.add_argument("--input", default="benchmark/results.csv",
                        help="Path to results.csv")
    parser.add_argument("--output", default="kernel_profile.png",
                        help="Output image path")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run bench_inference first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"benchmark", "variant", "parameter", "time_ms",
                "metric_value", "metric_unit"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns {missing} in {csv_path}",
              file=sys.stderr)
        sys.exit(1)

    df["parameter"] = df["parameter"].astype(str)

    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("KernelFlow - Kernel Performance Profile", fontsize=18,
                 fontweight="bold", color=GPU_GREEN, y=0.97)

    panel_gemm_scaling(axes[0, 0], df)
    panel_gemm_latency(axes[0, 1], df)
    panel_activations(axes[1, 0], df)
    panel_efficiency(axes[1, 1], df)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved -> {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
