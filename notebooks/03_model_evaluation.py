"""
03_model_evaluation.py
Evaluate trained YOLOv8 model on the TEST split: official metrics,
confusion matrix, per-class precision/recall/F1, and TP/FP/FN counts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish',
               'Gourami', 'MorishIdol', 'PlatyFish', 'RibbonedSweetlips',
               'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang', 'ZebraFish']

with open(PROJECT_ROOT / "outputs" / "training" / "best_weights_path.txt") as f:
    WEIGHTS_PATH = f.read().strip()

model = YOLO(WEIGHTS_PATH)

# ----------------------------------------------------------------
# 1. Run official validation on the TEST split
# ----------------------------------------------------------------
metrics = model.val(
    data=str(DATASET_DIR / "data.yaml"),
    split="test",
    conf=0.25,
    iou=0.5,
    save_json=True,
    plots=True,
    project=str(OUTPUT_DIR),
    name="test_eval"
)

# ----------------------------------------------------------------
# 2. Per-class precision / recall / mAP / F1
# ----------------------------------------------------------------
per_class = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": metrics.box.p,
    "Recall": metrics.box.r,
    "mAP50": metrics.box.ap50,
    "mAP50-95": metrics.box.ap,
    "F1": metrics.box.f1
})
per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
print(per_class)

overall = {
    "Precision(mean)": metrics.box.mp,
    "Recall(mean)": metrics.box.mr,
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map
}
pd.Series(overall).to_csv(OUTPUT_DIR / "overall_metrics.csv")
print(overall)

# ----------------------------------------------------------------
# 3. Confusion matrix
#    matrix shape = (nc+1, nc+1); last row/col = "background".
#    Ultralytics convention: rows = predicted class, cols = true class.
#    Verify against runs/.../test_eval/confusion_matrix.png (auto-saved)
#    if orientation ever looks off for your ultralytics version.
# ----------------------------------------------------------------
cm = metrics.confusion_matrix.matrix
labels = CLASS_NAMES + ["background"]

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.index.name = "Predicted"
cm_df.columns.name = "True"
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix_raw.csv")

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt=".0f", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# ----------------------------------------------------------------
# 4. TP / FP / FN per class derived from the confusion matrix
# ----------------------------------------------------------------
nc = len(CLASS_NAMES)
tp = np.diag(cm)[:nc]
fp = cm[:nc, :].sum(axis=1) - tp   # predicted as class i, but wrong (incl. bg mispredicted as class)
fn = cm[:, :nc].sum(axis=0) - tp   # true class i, but missed or mispredicted

tpfpfn = pd.DataFrame({"Class": CLASS_NAMES, "TP": tp, "FP": fp, "FN": fn})
tpfpfn["Precision"] = tpfpfn["TP"] / (tpfpfn["TP"] + tpfpfn["FP"]).replace(0, np.nan)
tpfpfn["Recall"] = tpfpfn["TP"] / (tpfpfn["TP"] + tpfpfn["FN"]).replace(0, np.nan)
tpfpfn.to_csv(OUTPUT_DIR / "tp_fp_fn_per_class.csv", index=False)
print(tpfpfn)

print("Evaluation complete. See outputs/evaluation/ and figures/confusion_matrix.png")
