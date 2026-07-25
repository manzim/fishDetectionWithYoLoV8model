"""
01_dataset_exploration.py
Dataset exploration for FishDetection.v5i.yolov11

Extends notebook 00 with: species frequency, fish-per-image distribution,
bounding-box geometry, image-size distribution, and class imbalance.
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

plt.style.use("ggplot")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exploration"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(PROJECT_ROOT)
print(DATASET_DIR)

# ---------------------------------------------------------
# Dataset Information
# ---------------------------------------------------------

CLASS_NAMES = [
    "AngelFish", "BlueTang", "ButterflyFish", "ClownFish", "GoldFish",
    "Gourami", "MorishIdol", "PlatyFish", "RibbonedSweetlips",
    "ThreeStripedDamselfish", "YellowCichlid", "YellowTang", "ZebraFish"
]

NUM_CLASSES = len(CLASS_NAMES)
print(f"Number of Classes : {NUM_CLASSES}")

# ---------------------------------------------------------
# Containers
# ---------------------------------------------------------

records = []
species_counter = Counter()
objects_per_image = []
bbox_widths = []
bbox_heights = []
bbox_areas = []
image_widths = []
image_heights = []

# ---------------------------------------------------------
# Read Entire Dataset
# ---------------------------------------------------------

for split in ["train", "valid", "test"]:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"
    if not label_dir.exists():
        label_dir = DATASET_DIR / split / "label"

    image_files = sorted(image_dir.glob("*.*"))

    print(f"\nReading {split} dataset...")
    print(f"Images : {len(image_files)}")

    for image_path in image_files:

        with Image.open(image_path) as img:
            width, height = img.size

        image_widths.append(width)
        image_heights.append(height)

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        lines = label_path.read_text().strip().splitlines()
        objects_per_image.append(len(lines))

        for line in lines:
            values = line.split()
            if len(values) != 5:
                continue

            cls = int(float(values[0]))
            x, y, bw, bh = map(float, values[1:])

            bbox_widths.append(bw)
            bbox_heights.append(bh)
            bbox_areas.append(bw * bh)

            species_counter[CLASS_NAMES[cls]] += 1

            records.append({
                "Split": split,
                "Image": image_path.name,
                "Class_ID": cls,
                "Species": CLASS_NAMES[cls],
                "Image_Width": width,
                "Image_Height": height,
                "Center_X": x,
                "Center_Y": y,
                "BBox_Width": bw,
                "BBox_Height": bh,
                "BBox_Area": bw * bh
            })

print("\nFinished reading dataset.")

df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "annotations_master.csv", index=False)
print(df.head())
print(df.shape)

# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

summary = {
    "Total Images": len(image_widths),
    "Total Fish Objects": len(df),
    "Number of Species": NUM_CLASSES,
    "Average Fish per Image": round(np.mean(objects_per_image), 2),
    "Median Fish per Image": np.median(objects_per_image),
    "Maximum Fish in One Image": np.max(objects_per_image),
    "Minimum Fish in One Image": np.min(objects_per_image),
    "Average Bounding Box Area": round(np.mean(bbox_areas), 5),
}

summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
summary_df.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
print(summary_df)

# ---------------------------------------------------------
# Species Frequency + Class Imbalance
# ---------------------------------------------------------

species_df = df["Species"].value_counts().reset_index()
species_df.columns = ["Species", "Fish_Count"]
species_df["Percentage"] = (species_df["Fish_Count"] / species_df["Fish_Count"].sum() * 100).round(2)

# Ensure every class appears even if count is zero
species_df = (
    pd.DataFrame({"Species": CLASS_NAMES})
    .merge(species_df, on="Species", how="left")
    .fillna({"Fish_Count": 0, "Percentage": 0.0})
)
species_df["Fish_Count"] = species_df["Fish_Count"].astype(int)
species_df = species_df.sort_values("Fish_Count", ascending=False).reset_index(drop=True)
species_df.to_csv(OUTPUT_DIR / "species_frequency.csv", index=False)

max_count = species_df["Fish_Count"].max()
min_count = species_df[species_df["Fish_Count"] > 0]["Fish_Count"].min()
imbalance_ratio = round(max_count / min_count, 2)

imbalance_summary = pd.DataFrame({
    "Metric": ["Most Frequent Species", "Most Frequent Count",
               "Least Frequent Species", "Least Frequent Count",
               "Imbalance Ratio (max/min)"],
    "Value": [
        species_df.iloc[0]["Species"], max_count,
        species_df[species_df["Fish_Count"] == min_count].iloc[0]["Species"], min_count,
        imbalance_ratio
    ]
})
imbalance_summary.to_csv(OUTPUT_DIR / "class_imbalance_summary.csv", index=False)
print(imbalance_summary)

# ---------------------------------------------------------
# Figure 1 - Species Frequency
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.bar(species_df["Species"], species_df["Fish_Count"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Fish Species")
plt.ylabel("Number of Annotated Fish")
plt.title("Species Frequency Distribution in the Fish Detection Dataset")
plt.tight_layout()
plt.savefig(FIG_DIR / "species_frequency.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2 - Species Percentage
# ---------------------------------------------------------

plt.figure(figsize=(10, 10))
plt.pie(species_df["Fish_Count"], labels=species_df["Species"], autopct="%1.1f%%", startangle=90)
plt.title("Relative Percentage of Fish Species")
plt.savefig(FIG_DIR / "species_percentage.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 3 - Objects per Image
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(objects_per_image, bins=range(0, max(objects_per_image) + 2))
plt.xlabel("Fish Objects per Image")
plt.ylabel("Number of Images")
plt.title("Distribution of Fish Count per Image")
plt.tight_layout()
plt.savefig(FIG_DIR / "objects_per_image.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 4 - BBox Width vs Height (scatter, size/shape of fish boxes)
# ---------------------------------------------------------

plt.figure(figsize=(7, 7))
plt.scatter(bbox_widths, bbox_heights, s=4, alpha=0.3)
plt.xlabel("Normalized Bounding Box Width")
plt.ylabel("Normalized Bounding Box Height")
plt.title("Bounding Box Width vs Height (Aspect Ratio Spread)")
plt.tight_layout()
plt.savefig(FIG_DIR / "bbox_width_vs_height.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 5 - BBox Area Distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(bbox_areas, bins=30)
plt.xlabel("Normalized Bounding Box Area")
plt.ylabel("Count")
plt.title("Bounding Box Area Distribution (small vs large fish)")
plt.tight_layout()
plt.savefig(FIG_DIR / "bbox_size_distribution.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 6 - Image Size Distribution (sanity check after resizing)
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(image_widths, image_heights, s=4, alpha=0.3)
plt.xlabel("Image Width (px)")
plt.ylabel("Image Height (px)")
plt.title("Image Dimension Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "image_size_distribution.png", dpi=300)
plt.close()

print("\nExploration completed. Summary and figures saved in outputs/exploration and figures/.")
