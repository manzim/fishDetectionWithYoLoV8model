"""
06_confidence_threshold_analysis.py
Compare detection performance at confidence thresholds 0.25, 0.50, 0.75.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "threshold_analysis"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "outputs" / "training" / "best_weights_path.txt") as f:
    WEIGHTS_PATH = f.read().strip()

model = YOLO(WEIGHTS_PATH)

THRESHOLDS = [0.25, 0.50, 0.75]
rows = []

for conf in THRESHOLDS:
    metrics = model.val(
        data=str(DATASET_DIR / "data.yaml"),
        split="test",
        conf=conf,
        iou=0.5,
        plots=False,
        project=str(OUTPUT_DIR),
        name=f"conf_{conf}"
    )

    cm = metrics.confusion_matrix.matrix
    nc = cm.shape[0] - 1
    tp = cm.diagonal()[:nc].sum()
    fp = cm[:nc, :].sum() - tp
    fn = cm[:, :nc].sum() - tp

    rows.append({
        "Confidence_Threshold": conf,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "TP": tp, "FP": fp, "FN": fn
    })

threshold_df = pd.DataFrame(rows)
threshold_df.to_csv(OUTPUT_DIR / "threshold_comparison.csv", index=False)
print(threshold_df)

# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(threshold_df["Confidence_Threshold"], threshold_df["Precision"], marker="o", label="Precision")
axes[0].plot(threshold_df["Confidence_Threshold"], threshold_df["Recall"], marker="o", label="Recall")
axes[0].plot(threshold_df["Confidence_Threshold"], threshold_df["mAP50"], marker="o", label="mAP50")
axes[0].set_xlabel("Confidence Threshold")
axes[0].set_title("Precision / Recall / mAP vs Confidence Threshold")
axes[0].legend()

width = 0.25
x = range(len(threshold_df))
axes[1].bar([i - width for i in x], threshold_df["TP"], width=width, label="TP")
axes[1].bar(x, threshold_df["FP"], width=width, label="FP")
axes[1].bar([i + width for i in x], threshold_df["FN"], width=width, label="FN")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(threshold_df["Confidence_Threshold"])
axes[1].set_xlabel("Confidence Threshold")
axes[1].set_title("TP / FP / FN vs Confidence Threshold")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "confidence_threshold_analysis.png", dpi=300)
plt.close()

print("Confidence threshold analysis complete.")
