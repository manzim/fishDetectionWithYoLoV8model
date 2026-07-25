"""
07_error_analysis.py
Quantitative failure analysis: matches predictions to ground truth via IoU,
then buckets errors into False Positives / False Negatives, and looks at
whether small objects drive more misses.

NOTE ON SCOPE: categories like "occlusion", "low lighting", "motion blur",
"schooling fish", and "background confusion" require visual judgment that
can't be reliably automated from bounding boxes alone. This script surfaces
the FP/FN examples (false_positives.csv / false_negatives.csv, plus saved
crops) so you can manually tag a sample of them for the qualitative part
of your error-analysis chapter.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis"
FIG_DIR = PROJECT_ROOT / "figures"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish',
               'Gourami', 'MorishIdol', 'PlatyFish', 'RibbonedSweetlips',
               'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang', 'ZebraFish']

with open(PROJECT_ROOT / "outputs" / "training" / "best_weights_path.txt") as f:
    WEIGHTS_PATH = f.read().strip()

model = YOLO(WEIGHTS_PATH)

TEST_IMG_DIR = DATASET_DIR / "test" / "images"
TEST_LBL_DIR = DATASET_DIR / "test" / "labels"
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5
SMALL_BOX_AREA = 0.01     # normalized area cutoff for "small" fish
MAX_SAVED_CROPS = 30      # cap example images saved to disk


def yolo_to_xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


fp_records, fn_records = [], []
n_fp_saved, n_fn_saved = 0, 0

image_paths = sorted(TEST_IMG_DIR.glob("*.*"))

for img_path in image_paths:
    label_path = TEST_LBL_DIR / f"{img_path.stem}.txt"
    gt_boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0]))
            x, y, w, h = map(float, p[1:])
            gt_boxes.append({"cls": cid, "box": yolo_to_xyxy(x, y, w, h), "area": w * h})

    result = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, cid, conf in zip(result.boxes.xyxyn.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy().astype(int),
                                   result.boxes.conf.cpu().numpy()):
            pred_boxes.append({"cls": cid, "box": tuple(box), "conf": float(conf)})

    matched_gt, matched_pred = set(), set()

    for pi, pred in enumerate(pred_boxes):
        best_iou, best_gi = 0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt or gt["cls"] != pred["cls"]:
                continue
            cur_iou = iou(pred["box"], gt["box"])
            if cur_iou > best_iou:
                best_iou, best_gi = cur_iou, gi
        if best_iou >= IOU_MATCH_THRESHOLD:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    # False Positives: predicted boxes with no matching ground truth
    for pi, pred in enumerate(pred_boxes):
        if pi not in matched_pred:
            fp_records.append({
                "Image": img_path.name,
                "Predicted_Class": CLASS_NAMES[pred["cls"]],
                "Confidence": pred["conf"]
            })

            if n_fp_saved < MAX_SAVED_CROPS:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = pred["box"]
                    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)
                    cv2.imwrite(str(EXAMPLES_DIR / f"FP_{img_path.stem}_{pi}.jpg"), img)
                    n_fp_saved += 1

    # False Negatives: ground-truth boxes with no matching prediction
    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt:
            size_cat = "small" if gt["area"] < SMALL_BOX_AREA else "normal"
            fn_records.append({
                "Image": img_path.name,
                "True_Class": CLASS_NAMES[gt["cls"]],
                "Box_Area": gt["area"],
                "Size_Category": size_cat
            })

            if n_fn_saved < MAX_SAVED_CROPS:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = gt["box"]
                    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2)
                    cv2.imwrite(str(EXAMPLES_DIR / f"FN_{img_path.stem}_{gi}.jpg"), img)
                    n_fn_saved += 1

fp_df = pd.DataFrame(fp_records)
fn_df = pd.DataFrame(fn_records)
fp_df.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
fn_df.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)

# ----------------------------------------------------------------
# Summary stats
# ----------------------------------------------------------------
fn_by_size = fn_df["Size_Category"].value_counts() if not fn_df.empty else pd.Series(dtype=int)
fn_by_size.to_csv(OUTPUT_DIR / "fn_by_size_category.csv")

fp_by_class = fp_df["Predicted_Class"].value_counts() if not fp_df.empty else pd.Series(dtype=int)
fn_by_class = fn_df["True_Class"].value_counts() if not fn_df.empty else pd.Series(dtype=int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if not fp_by_class.empty:
    fp_by_class.plot(kind="bar", ax=axes[0], color="salmon")
axes[0].set_title("False Positives by Predicted Class")
if not fn_by_class.empty:
    fn_by_class.plot(kind="bar", ax=axes[1], color="indianred")
axes[1].set_title("False Negatives by True Class")
plt.tight_layout()
plt.savefig(FIG_DIR / "error_by_class.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
if not fn_by_size.empty:
    fn_by_size.plot(kind="bar", color="darkred")
plt.title("False Negatives by Object Size (small vs normal)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fn_by_size.png", dpi=300)
plt.close()

print(f"False Positives: {len(fp_df)}")
print(f"False Negatives: {len(fn_df)}")
print(fn_by_size)
print(f"\nExample crops saved to {EXAMPLES_DIR} ({n_fp_saved} FP, {n_fn_saved} FN)")
print("Review these manually to tag qualitative categories (occlusion, blur,")
print("low light, schooling, background confusion, class confusion) for your report.")
