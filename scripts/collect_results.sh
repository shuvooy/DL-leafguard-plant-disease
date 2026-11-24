#!/usr/bin/env bash
set -euo pipefail

WORK="${LEAFGUARD_WORK:-$HOME/Documents/AI Engineering/plant disease project/leafguard_output}"
RUN="${1:-convnext_tiny}"
LAB="${2:-lab_only_convnext_tiny}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EVAL="$WORK/runs/$RUN/eval"

mkdir -p "$REPO/results/lab_only"

for f in confusion_test.png confusion_plantdoc_test.png roc_pr_test.png reliability.png risk_coverage.png \
         robustness.png gradcam.png confident_mistakes.png worst_classes_test.png worst_classes_plantdoc_test.png \
         report.md metrics.json calibration.json per_class_test.csv per_class_plantdoc_test.csv \
         top_confusions_test.csv top_confusions_plantdoc_test.csv; do
  [ -f "$EVAL/$f" ] && cp "$EVAL/$f" "$REPO/results/$f" || echo "missing: $EVAL/$f"
done

for f in training_curves.png history.csv run_info.json; do
  [ -f "$WORK/runs/$RUN/$f" ] && cp "$WORK/runs/$RUN/$f" "$REPO/results/$f" || echo "missing: $WORK/runs/$RUN/$f"
done

for f in class_distribution.png data_report.json label_map.csv comparison.md; do
  [ -f "$WORK/$f" ] && cp "$WORK/$f" "$REPO/results/$f" || echo "missing: $WORK/$f"
done

for f in report.md metrics.json reliability.png; do
  [ -f "$WORK/runs/$LAB/eval/$f" ] && cp "$WORK/runs/$LAB/eval/$f" "$REPO/results/lab_only/$f" || echo "missing: lab_only/$f"
done
[ -f "$WORK/runs/$LAB/history.csv" ] && cp "$WORK/runs/$LAB/history.csv" "$REPO/results/lab_only/history.csv" || true

DEPLOY="${LEAFGUARD_DEPLOY_SRC:-$HOME/Documents/AI Engineering/plant disease project/deploy}"
mkdir -p "$REPO/deploy/samples"
for f in model_config.json model_int8.onnx; do
  [ -f "$DEPLOY/$f" ] && cp "$DEPLOY/$f" "$REPO/deploy/$f" || echo "missing: $DEPLOY/$f"
done
cp "$DEPLOY"/samples/*.jpg "$REPO/deploy/samples/" 2>/dev/null || echo "missing: deploy samples"

echo "done. fp32 model to upload as a GitHub Release asset: $DEPLOY/model.onnx"
