#!/usr/bin/env python3
"""Rebuild the small summary charts from the curated project metrics JSON.

The charts visualize summary values documented in docs/RESULTS.md. They are
not substitutes for per-image evaluation artifacts or raw prediction files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = json.loads((RESULTS / "summary_metrics.json").read_text(encoding="utf-8"))


def finish(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def accuracy_summary() -> None:
    data = METRICS["accuracy_percent"]
    labels = ["Internal test\n(all)", "Internal test\n(lab)", "Internal test\n(field)", "PlantDoc\n(held out)"]
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("LeafGuard evaluation accuracy")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    ax.text(0.01, -0.22, "Different evaluation subsets; see docs/RESULTS.md for sample counts and caveats.",
            transform=ax.transAxes, fontsize=8)
    finish(fig, "accuracy_summary.png")


def domain_gap() -> None:
    data = METRICS["plantdoc_domain_gap_percent"]
    labels = ["Lab-only\n(PlantVillage)", "Lab + field\n(main model)"]
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 80)
    ax.set_ylabel("PlantDoc held-out accuracy (%)")
    ax.set_title("Training with field photos reduces the domain gap")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    ax.text(0.01, -0.19, "Both models evaluated on the same 223 PlantDoc photos.",
            transform=ax.transAxes, fontsize=8)
    finish(fig, "plantdoc_domain_gap.png")


def robustness() -> None:
    data = METRICS["robustness_accuracy_percent"]
    labels = [
        "Clean", "Low contrast", "Dark", "Rotate 30°", "Noise σ=15", "Bright",
        "Blur r=2", "JPEG q=15", "Low resolution", "Occlusion 25%", "Noise σ=35", "Blur r=5"
    ]
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Robustness under 12 photo conditions")
    ax.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, values):
        ax.text(value + 1, yi, f"{value:.1f}%", va="center", fontsize=8)
    ax.text(0.01, -0.12, "1,500 internal test photos; strong blur is the largest failure mode.",
            transform=ax.transAxes, fontsize=8)
    finish(fig, "robustness_summary.png")


def calibration() -> None:
    data = METRICS["ece"]
    sets = ["Validation", "Internal test", "PlantDoc"]
    before = [data["validation_before"], data["test_before"], data["plantdoc_before"]]
    after = [data["validation_after"], data["test_after"], data["plantdoc_after"]]
    x = np.arange(len(sets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    a = ax.bar(x - width / 2, before, width, label="Before temperature scaling")
    b = ax.bar(x + width / 2, after, width, label="After temperature scaling")
    ax.set_xticks(x, sets)
    ax.set_ylabel("Expected calibration error (percentage points)")
    ax.set_title("Calibration improves in-domain, but shifts on PlantDoc")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in (a, b):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)
    finish(fig, "calibration_summary.png")


def selective_prediction() -> None:
    data = METRICS["selective_prediction"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    points = [
        (data["internal_test_coverage_percent"], data["internal_test_accuracy_percent"], "Internal test"),
        (data["plantdoc_coverage_percent"], data["plantdoc_accuracy_percent"], "PlantDoc"),
    ]
    for x, y, label in points:
        ax.scatter(x, y, s=90, label=label)
        ax.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points")
    ax.axhline(data["target_accuracy_percent"], linestyle="--", label="95% target")
    ax.set_xlim(0, 100)
    ax.set_ylim(70, 101)
    ax.set_xlabel("Share of photos answered (%)")
    ax.set_ylabel("Accuracy on answered photos (%)")
    ax.set_title("Selective prediction: coverage vs accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, "selective_prediction.png")


def latency() -> None:
    data = METRICS["cpu_latency_ms"]
    labels = ["FP32 ONNX", "INT8 ONNX"]
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("CPU latency per photo (ms, 4 threads)")
    ax.set_title("Compact does not mean faster in this export")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f} ms", ha="center")
    ax.text(0.01, -0.18,
            f"INT8 model: {METRICS['int8_prediction_agreement_percent']:.2f}% prediction agreement with FP32.",
            transform=ax.transAxes, fontsize=8)
    finish(fig, "cpu_latency_summary.png")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    accuracy_summary()
    domain_gap()
    robustness()
    calibration()
    selective_prediction()
    latency()
    print("Built 6 summary charts in", RESULTS)


if __name__ == "__main__":
    main()
