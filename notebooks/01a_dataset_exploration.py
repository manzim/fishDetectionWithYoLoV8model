from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

plt.style.use("ggplot")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# PROJECT_ROOT = Path.cwd().parent

# DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"

# OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exploration"

# FIGURE_DIR = PROJECT_ROOT / "figures"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print(PROJECT_ROOT)
print(DATASET_DIR)

# ---------------------------------------------------------
# Dataset Information
# ---------------------------------------------------------

CLASS_NAMES = [

    "AngelFish",
    "BlueTang",
    "ButterflyFish",
    "ClownFish",
    "GoldFish",
    "Gourami",
    "MorishIdol",
    "PlatyFish",
    "RibbonedSweetlips",
    "ThreeStripedDamselfish",
    "YellowCichlid",
    "YellowTang",
    "ZebraFish"
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

            cls = int(values[0])

            x = float(values[1])

            y = float(values[2])

            bw = float(values[3])

            bh = float(values[4])

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

# ---------------------------------------------------------
# Master DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(records)

print(df.head())

print()

print(df.shape)

# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

summary = {

    "Total Images": len(image_widths),

    "Total Fish Objects": len(df),

    "Number of Species": NUM_CLASSES,

    "Average Fish per Image":
        round(np.mean(objects_per_image),2),

    "Median Fish per Image":
        np.median(objects_per_image),

    "Maximum Fish in One Image":
        np.max(objects_per_image),

    "Minimum Fish in One Image":
        np.min(objects_per_image),

    "Average Bounding Box Area":
        round(np.mean(bbox_areas),5)

}

summary_df = pd.DataFrame(

    summary.items(),

    columns=["Metric","Value"]

)

summary_df

# Save Summary

summary_df.to_csv(

    OUTPUT_DIR / "dataset_summary.csv",

    index=False

)

# ---------------------------------------------------------
# Species Frequency
# ---------------------------------------------------------

species_df = (

    df["Species"]

    .value_counts()

    .reset_index()

)

species_df.columns = [

    "Species",

    "Fish_Count"

]

species_df["Percentage"] = (

    species_df["Fish_Count"]

    / species_df["Fish_Count"].sum()

    *100

).round(2)

species_df

species_df.to_csv(

    OUTPUT_DIR / "species_frequency.csv",

    index=False

)

# ---------------------------------------------------------
# Figure 1
# Species Frequency
# ---------------------------------------------------------

plt.figure(figsize=(12,6))

plt.bar(

    species_df["Species"],

    species_df["Fish_Count"]

)

plt.xticks(

    rotation=45,

    ha="right"

)

plt.xlabel("Fish Species")

plt.ylabel("Number of Annotated Fish")

plt.title("Species Frequency Distribution in the Fish Detection Dataset")

plt.tight_layout()

plt.savefig(

    FIG_DIR /

    "species_frequency.png",

    dpi=300

)

plt.show()

# ---------------------------------------------------------
# Figure 2
# Species Percentage
# ---------------------------------------------------------

plt.figure(figsize=(10,10))

plt.pie(

    species_df["Fish_Count"],

    labels=species_df["Species"],

    autopct="%1.1f%%",

    startangle=90

)

plt.title(

    "Relative Percentage of Fish Species"

)

plt.savefig(

    FIG_DIR /

    "species_percentage.png",

    dpi=300

)

plt.show()

print ("Exploration completed. Summary and figures saved in the outputs and figures directories.")