### 02_train_yolov8.py



from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
FIGURE_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_YAML.exists(), f"Cannot find {DATA_YAML}"

print(f"Dataset : {DATASET_DIR}")
print(f"Using   : {DATA_YAML}")


DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("-" * 60)
print("Hardware")
print("-" * 60)

if DEVICE == 0:
    print("GPU detected")
    print(torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")
    print("Training will run on CPU (slower).")

print("-" * 60)

# ==========================================================
# Training Configuration
# ==========================================================

MODEL = "yolov8n.pt"
EPOCHS = 40
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 10
RUN_NAME = "YOLOv8n_40epochs"

# ==========================================================
# Load model
# ==========================================================

model = YOLO(MODEL)
print("\nStarting training...\n")
results = model.train(

    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=PATIENCE,
    project=str(RUNS_DIR),
    name=RUN_NAME,
    cache=False,
    workers=0,
    seed=42,
    deterministic=True,
    verbose=True,
    plots=True,
    save_period=1
)

# ==========================================================
# Training finished
# ==========================================================

run_dir = Path(results.save_dir)
print("\nTraining completed.")
print("Run directory:")
print(run_dir)

# ==========================================================
# Save best model path
# ==========================================================

best_model = run_dir / "weights" / "best.pt"

with open(OUTPUT_DIR / "best_weights_path.txt", "w") as f:
    f.write(str(best_model))
print("\nBest model:")
print(best_model)

# ==========================================================
# Save prediction examples
# ==========================================================

print("\nGenerating prediction examples...")
model = YOLO(best_model)
model.predict(
    source=str(DATASET_DIR / "test" / "images"),
    conf=0.25,
    save=True,
    project=str(RUNS_DIR),
    name="test_predictions"
)

# ==========================================================
# Read training statistics
# ==========================================================

results_csv = run_dir / "results.csv"
df = pd.read_csv(results_csv)
df.columns = [c.strip() for c in df.columns]
df.to_csv(
    OUTPUT_DIR / "training_results.csv",
    index=False
)

print("\nLast training epochs:\n")
print(df.tail())

# ==========================================================
# Plot training curves
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(df["epoch"], df["train/box_loss"], label="Train")
axes[0,0].plot(df["epoch"], df["val/box_loss"], label="Validation")
axes[0,0].set_title("Bounding Box Loss")
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Loss")
axes[0,0].legend()

# ----------------------------------------------------------

axes[0,1].plot(df["epoch"], df["train/cls_loss"], label="Train")
axes[0,1].plot(df["epoch"], df["val/cls_loss"], label="Validation")
axes[0,1].set_title("Classification Loss")
axes[0,1].set_xlabel("Epoch")
axes[0,1].set_ylabel("Loss")
axes[0,1].legend()

# ----------------------------------------------------------

axes[1,0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
axes[1,0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
axes[1,0].set_title("Precision and Recall")
axes[1,0].set_xlabel("Epoch")
axes[1,0].set_ylabel("Score")
axes[1,0].legend()

# ----------------------------------------------------------

axes[1,1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.50")
axes[1,1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.50:0.95")
axes[1,1].set_title("Mean Average Precision")
axes[1,1].set_xlabel("Epoch")
axes[1,1].set_ylabel("Score")
axes[1,1].legend()

# ----------------------------------------------------------

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "training_curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nTraining curves saved.")

# ==========================================================
# Final summary
# ==========================================================

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Model            : {MODEL}")
print(f"Epochs           : {len(df)}")
print(f"Image Size       : {IMAGE_SIZE}")
print(f"Best Model       : {best_model}")
print(f"Training Results : {OUTPUT_DIR/'training_results.csv'}")
print(f"Training Figure  : {FIGURE_DIR/'training_curves.png'}")
print("="*60)