"""
04_fish_counting.py
Run inference on every test image and produce per-image species counts.
Feeds directly into notebook 05 (community composition).
"""

from pathlib import Path
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "counting"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['AngelFish', 'BlueTang', 'ButterflyFish', 'ClownFish', 'GoldFish',
               'Gourami', 'MorishIdol', 'PlatyFish', 'RibbonedSweetlips',
               'ThreeStripedDamselfish', 'YellowCichlid', 'YellowTang', 'ZebraFish']

with open(PROJECT_ROOT / "outputs" / "training" / "best_weights_path.txt") as f:
    WEIGHTS_PATH = f.read().strip()

model = YOLO(WEIGHTS_PATH)

TEST_IMG_DIR = DATASET_DIR / "test" / "images"
CONF_THRESHOLD = 0.25

per_image_rows = []
detection_records = []

# save=True writes annotated images under outputs/counting/predictions/
results = model.predict(
    source=str(TEST_IMG_DIR),
    conf=CONF_THRESHOLD,
    save=True,
    project=str(OUTPUT_DIR),
    name="predictions",
    verbose=False
)

for r in results:
    img_name = Path(r.path).name

    if r.boxes is not None and len(r.boxes) > 0:
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
    else:
        class_ids, confs = [], []

    counts = {name: 0 for name in CLASS_NAMES}
    for cid in class_ids:
        counts[CLASS_NAMES[cid]] += 1

    row = {"Image": img_name, "Total_Fish": len(class_ids)}
    row.update(counts)
    per_image_rows.append(row)

    for cid, conf in zip(class_ids, confs):
        detection_records.append({
            "Image": img_name,
            "Species": CLASS_NAMES[cid],
            "Confidence": float(conf)
        })

per_image_df = pd.DataFrame(per_image_rows)
per_image_df.to_csv(OUTPUT_DIR / "fish_counts_per_image.csv", index=False)

detections_df = pd.DataFrame(detection_records)
detections_df.to_csv(OUTPUT_DIR / "all_detections.csv", index=False)

species_totals = (
    detections_df["Species"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    .reset_index()
)
species_totals.columns = ["Species", "Total_Count"]
species_totals.to_csv(OUTPUT_DIR / "species_totals.csv", index=False)

print(per_image_df.head())
print(species_totals)
print(f"\nTotal images processed: {len(per_image_df)}")
print(f"Total fish detected: {detections_df.shape[0]}")
print("Fish counting complete. Annotated images saved in outputs/counting/predictions/.")
