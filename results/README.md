# Evaluation summary assets

This directory contains:

- `summary_metrics.json`: the curated numeric values used for the summary charts.
- `accuracy_summary.png`, `plantdoc_domain_gap.png`, `robustness_summary.png`, `calibration_summary.png`, `selective_prediction.png`, and `cpu_latency_summary.png`: visualizations recreated from the summary values in [`../docs/RESULTS.md`](../docs/RESULTS.md) and the exported model configuration.

These plots are **summary-level visualizations**, not the original per-image evaluation output or raw prediction arrays. The original generated reports, per-class CSV exports, and training history are not included in this source package. Rebuild the charts with:

```bash
python scripts/build_summary_charts.py
```

For the methodology, sample counts, full reported tables, and limitations, see [`../docs/RESULTS.md`](../docs/RESULTS.md).
