"""
KernelFlow — RL Reward & Epsilon Visualization
================================================
Reads rewards.csv (columns: episode, avg_reward, epsilon) and produces
reward_curve.png with two subplots:
  1. Reward curve with raw + smoothed line, solve threshold marker
  2. Epsilon decay curve over training

Usage:
    python viz/reward_plot.py                          # default path
    python viz/reward_plot.py --input path/to/rewards.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
SOLVE_THRESHOLD = 195.0
SMOOTH_WINDOW = 20         # rolling‑mean window for reward smoothing

# ── Professional dark theme colours ──────────────────────────────────────────
BG_DARK = "#1a1a2e"
BG_AXES = "#16213e"
GPU_GREEN = "#76b900"      # NVIDIA green
GPU_GREEN_LIGHT = "#a3d94e"
EPSILON_CYAN = "#00d4ff"
THRESHOLD_RED = "#ff4c4c"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that preserves array length (edges use partial windows)."""
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KernelFlow RL training curves")
    parser.add_argument("--input", default="rewards.csv", help="Path to rewards.csv")
    parser.add_argument("--output", default="reward_curve.png", help="Output image path")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run DQN training first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    for col in ("episode", "avg_reward", "epsilon"):
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}' in {csv_path}", file=sys.stderr)
            sys.exit(1)

    episodes = df["episode"].values
    rewards = df["avg_reward"].values
    epsilons = df["epsilon"].values
    smoothed = smooth(rewards, SMOOTH_WINDOW)

    # Find solve point (first episode where smoothed reward exceeds threshold)
    solve_idx = np.where(smoothed >= SOLVE_THRESHOLD)[0]
    solved = len(solve_idx) > 0
    solve_ep = episodes[solve_idx[0]] if solved else None

    # ── Set up figure ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "font.family": "monospace",
        "font.size": 11,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                   sharex=True, gridspec_kw={"hspace": 0.08})
    fig.suptitle("KernelFlow — DQN Training Progress", fontsize=16,
                 fontweight="bold", color=GPU_GREEN, y=0.96)

    # ── Plot 1: Reward curve ─────────────────────────────────────────────────
    ax1.fill_between(episodes, rewards, alpha=0.12, color=GPU_GREEN)
    ax1.plot(episodes, rewards, color=GPU_GREEN, alpha=0.25, linewidth=0.7,
             label="Raw reward")
    ax1.plot(episodes, smoothed, color=GPU_GREEN_LIGHT, linewidth=2.2,
             label=f"Smoothed (window={SMOOTH_WINDOW})")

    # Solve threshold line
    ax1.axhline(y=SOLVE_THRESHOLD, color=THRESHOLD_RED, linestyle="--",
                linewidth=1.2, alpha=0.7, label=f"Solve threshold ({SOLVE_THRESHOLD})")

    # Mark solve point
    if solved:
        ax1.axvline(x=solve_ep, color=THRESHOLD_RED, linestyle=":", linewidth=1, alpha=0.5)
        ax1.scatter([solve_ep], [smoothed[solve_idx[0]]], s=120, zorder=5,
                    color=THRESHOLD_RED, edgecolors="white", linewidths=1.5)
        ax1.annotate(f"Solved @ ep {solve_ep}",
                     xy=(solve_ep, smoothed[solve_idx[0]]),
                     xytext=(solve_ep + len(episodes) * 0.05,
                             smoothed[solve_idx[0]] * 0.92),
                     fontsize=10, color=THRESHOLD_RED, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=THRESHOLD_RED, lw=1.3))

    ax1.set_ylabel("Average Reward", fontsize=12)
    ax1.legend(loc="lower right", framealpha=0.3, edgecolor=GRID_COLOR)
    ax1.grid(True, linestyle="--", linewidth=0.5)
    ax1.set_xlim(episodes[0], episodes[-1])

    # ── Plot 2: Epsilon decay ────────────────────────────────────────────────
    ax2.fill_between(episodes, epsilons, alpha=0.15, color=EPSILON_CYAN)
    ax2.plot(episodes, epsilons, color=EPSILON_CYAN, linewidth=1.8, label="Epsilon")
    ax2.set_ylabel("Epsilon (ε)", fontsize=12)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylim(-0.02, max(epsilons) * 1.08)
    ax2.legend(loc="upper right", framealpha=0.3, edgecolor=GRID_COLOR)
    ax2.grid(True, linestyle="--", linewidth=0.5)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    # Avoid Unicode characters in console output on Windows
    print(f"Saved -> {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
