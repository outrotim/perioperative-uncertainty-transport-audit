"""Redraw the three main figures from non-identifiable aggregate data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "main_figure_data.json"
OUTPUT_DIR = ROOT / "figures"
COLORS = {"random": "#0072B2", "temporal": "#D55E00"}


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_1(data: dict) -> None:
    panel = data["figure_1"]
    cohort = panel["cohort"]
    timeline = panel["timeline"]
    fig, (flow_ax, time_ax) = plt.subplots(
        1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [0.86, 1.55]}
    )
    fig.suptitle(
        "Cohort construction, data provenance, and forward-in-time evaluation",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

    flow_ax.axis("off")
    box = dict(boxstyle="round,pad=0.38", linewidth=1.2)
    flow_ax.text(
        0.5,
        0.89,
        f"Linked hospitalizations\nn = {cohort['linked']:,}",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox={**box, "facecolor": "#D9EAF7", "edgecolor": "#0072B2"},
    )
    flow_ax.text(
        0.5,
        0.7,
        f"Both pain outcomes observed\nn = {cohort['eligible']:,} ({cohort['eligible_pct']:.1f}%)",
        ha="center",
        va="center",
        fontsize=10,
        bbox={**box, "facecolor": "#DDF2E9", "edgecolor": "#009E73"},
    )
    flow_ax.annotate(
        "", xy=(0.5, 0.78), xytext=(0.5, 0.82),
        arrowprops=dict(arrowstyle="->", color="#555555")
    )
    flow_ax.text(
        0.5,
        0.58,
        f"Excluded: {cohort['excluded']:,} ({cohort['excluded_pct']:.1f}%) "
        "with both outcomes missing",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#7A3E00",
    )
    coordinates = [(0.2, 0.39), (0.67, 0.39), (0.2, 0.18), (0.67, 0.18)]
    for (label, count), (x, y) in zip(cohort["splits"].items(), coordinates):
        flow_ax.text(
            x,
            y,
            f"{label}\n{count:,}",
            ha="center",
            va="center",
            fontsize=9.2,
            bbox={**box, "facecolor": "#F4F4F4", "edgecolor": "#777777"},
        )
        flow_ax.annotate(
            "", xy=(x, y + 0.08), xytext=(0.5, 0.5),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.9)
        )
    flow_ax.text(
        0.02, 0.98, "A  Cohort and frozen split",
        transform=flow_ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold"
    )
    flow_ax.text(
        0.02, 0.03,
        "Observed outcomes only; no target\nimputation or winsorisation.",
        transform=flow_ax.transAxes, ha="left", va="bottom", fontsize=8.8
    )

    xmin = np.datetime64(timeline["axis_start"])
    xmax = np.datetime64(timeline["axis_end"])
    time_ax.set_xlim(xmin, xmax)
    time_ax.set_ylim(0.55, 5.45)
    rows = [
        (4.75, "Earlier date provenance", "#56B4E9"),
        (3.90, "Later date provenance", "#E69F00"),
        (3.05, "Observed pain outcomes", "#009E73"),
        (2.05, "Development period", "#0072B2"),
        (1.20, "Forward temporal test", "#D55E00"),
    ]
    for y, label, color in rows:
        item = timeline["ranges"][label]
        start = np.datetime64(item["start"])
        end = np.datetime64(item["end"])
        time_ax.barh(
            y,
            (end - start).astype("timedelta64[D]").astype(int),
            left=start,
            height=0.38,
            color=color,
            alpha=0.88,
        )
        time_ax.text(
            xmin - np.timedelta64(25, "D"), y, label,
            ha="right", va="center", fontsize=9
        )
    for boundary in timeline["boundaries"]:
        date = np.datetime64(boundary["date"])
        time_ax.axvline(date, color="#555555", linewidth=0.8, linestyle="--")
        time_ax.text(
            date, 0.64, boundary["label"], rotation=90,
            ha="right", va="bottom", fontsize=7.8, color="#4D4D4D"
        )
    time_ax.xaxis.set_major_locator(mdates.YearLocator())
    time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    time_ax.set_yticks([])
    time_ax.set_xlabel("Calendar time")
    time_ax.set_title(
        "B  Timeline, date provenance, and forward transport",
        loc="left", fontsize=11, fontweight="bold"
    )
    time_ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "Figure_1_cohort_timeline_and_provenance")


def figure_2(data: dict) -> None:
    rows = data["figure_2"]["metrics"]
    specs = [
        ("r2", "A  Point prediction: $R^2$", 0.0),
        ("coverage_90", "B  90% normalized conformal coverage", 0.9),
        ("rho_uncertainty_abs_error", "C  Uncertainty–error Spearman $\\rho$", 0.0),
        ("delta_mae_retention_80", "D  $\\Delta$MAE at 80% retention", 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    y_positions = {"rest": 1, "activity": 0}
    offsets = {"random": 0.08, "temporal": -0.08}
    markers = {"random": "o", "temporal": "s"}

    for ax, (metric, title, reference) in zip(axes.flat, specs):
        selected = [row for row in rows if row["metric"] == metric]
        for task in ("rest", "activity"):
            task_rows = {row["evaluation"]: row for row in selected if row["task"] == task}
            if set(task_rows) == {"random", "temporal"}:
                ax.plot(
                    [task_rows["random"]["estimate"], task_rows["temporal"]["estimate"]],
                    [y_positions[task] + offsets["random"], y_positions[task] + offsets["temporal"]],
                    color="#B0B0B0",
                    linewidth=0.7,
                    zorder=1,
                )
            for evaluation in ("random", "temporal"):
                row = task_rows[evaluation]
                estimate = row["estimate"]
                ax.errorbar(
                    estimate,
                    y_positions[task] + offsets[evaluation],
                    xerr=[[estimate - row["ci95_low"]], [row["ci95_high"] - estimate]],
                    fmt=markers[evaluation],
                    color=COLORS[evaluation],
                    capsize=3,
                    markersize=5,
                    label=evaluation.capitalize() if task == "rest" else None,
                    zorder=2,
                )
        ax.axvline(reference, color="#444444", linewidth=0.9, linestyle="--")
        ax.set_yticks([0, 1], ["Activity pain", "Resting pain"])
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
        if metric == "coverage_90":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.text(
        0.05, 0.015,
        "Points are estimates; error bars are conditional bootstrap 95% CIs. "
        "For ΔMAE, negative values indicate lower retained-set error.",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, "Figure_2_random_vs_temporal_reliability")


def figure_3(data: dict) -> None:
    panel = data["figure_3"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), sharex=True, sharey=True)
    panels = [
        ("full", "rest", "A  Full model — resting pain"),
        ("full", "activity", "B  Full model — activity pain"),
        ("source_common", "rest", "C  Source-common — resting pain"),
        ("source_common", "activity", "D  Source-common — activity pain"),
    ]
    for ax, (model, task, title) in zip(axes.flat, panels):
        for evaluation in ("random", "temporal"):
            curve = next(
                item for item in panel["curves"]
                if item["model"] == model
                and item["task"] == task
                and item["evaluation"] == evaluation
            )
            ax.plot(
                curve["retained_percent"],
                curve["delta_mae"],
                color=COLORS[evaluation],
                linewidth=2,
                label=evaluation.capitalize(),
            )
            summary = curve["retention_80"]
            ax.errorbar(
                80,
                summary["estimate"],
                yerr=[
                    [summary["estimate"] - summary["ci95_low"]],
                    [summary["ci95_high"] - summary["estimate"]],
                ],
                fmt="o" if evaluation == "random" else "s",
                color=COLORS[evaluation],
                capsize=3,
                markersize=4.5,
            )
        ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
        ax.axvline(80, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2)
    for ax in axes[1, :]:
        ax.set_xlabel("Retained cases (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Change in retained-set MAE")
    fig.text(
        0.06, 0.015,
        "Negative values indicate lower MAE after deferring higher-uncertainty records; "
        "markers show 80% retention with conditional bootstrap 95% CIs.",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "Figure_3_risk_retention_curves")


def main() -> None:
    data = load_data()
    figure_1(data)
    figure_2(data)
    figure_3(data)
    print(f"Wrote main figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

