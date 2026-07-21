"""
00_dataset_validation.py
Dataset validation for FishDetection.v5i.yolov11
"""

from pathlib import Path
import hashlib
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "FishDetection.v5i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['AngelFish','BlueTang','ButterflyFish','ClownFish','GoldFish',
'Gourami','MorishIdol','PlatyFish','RibbonedSweetlips',
'ThreeStripedDamselfish','YellowCichlid','YellowTang','ZebraFish']

summary=[]
class_counts={i:0 for i in range(len(CLASS_NAMES))}
obj_per_image=[]
bbox_sizes=[]
duplicates={}

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""):
            h.update(c)
    return h.hexdigest()

for split in ["train","valid","test"]:
    img_dir=DATASET_DIR/split/"images"
    lbl_dir=DATASET_DIR/split/"labels"
    if not lbl_dir.exists():
        lbl_dir=DATASET_DIR/split/"label"

    imgs=list(img_dir.glob("*.*"))
    lbls=list(lbl_dir.glob("*.txt"))

    missing_labels=[]
    empty_labels=[]
    corrupt=[]
    invalid_boxes=0

    for img in imgs:
        duplicates.setdefault(md5(img),[]).append(str(img.name))
        try:
            Image.open(img).verify()
        except:
            corrupt.append(img.name)

        label=lbl_dir/(img.stem+".txt")
        if not label.exists():
            missing_labels.append(img.name)
            continue

        txt=label.read_text().strip()
        if txt=="":
            empty_labels.append(label.name)
            obj_per_image.append(0)
            continue

        lines=txt.splitlines()
        obj_per_image.append(len(lines))

        for ln in lines:
            p=ln.split()
            if len(p)!=5:
                invalid_boxes+=1
                continue
            c=int(float(p[0]))
            if c in class_counts:
                class_counts[c]+=1
            x,y,w,h=map(float,p[1:])
            if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                invalid_boxes+=1
            bbox_sizes.append(w*h)

    summary.append({
        "split":split,
        "images":len(imgs),
        "labels":len(lbls),
        "missing_labels":len(missing_labels),
        "empty_labels":len(empty_labels),
        "corrupted_images":len(corrupt),
        "invalid_boxes":invalid_boxes
    })

summary_df=pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_DIR/"dataset_summary.csv",index=False)

cls_df=pd.DataFrame({
    "Class":CLASS_NAMES,
    "Objects":[class_counts[i] for i in range(len(CLASS_NAMES))]
})
cls_df.to_csv(OUTPUT_DIR/"class_distribution.csv",index=False)

dup_rows=[]
for k,v in duplicates.items():
    if len(v)>1:
        dup_rows.append({"md5":k,"count":len(v),"files":"; ".join(v)})
pd.DataFrame(dup_rows).to_csv(OUTPUT_DIR/"duplicate_images.csv",index=False)

with open(OUTPUT_DIR/"validation_report.txt","w") as f:
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    f.write(cls_df.to_string(index=False))

plt.figure(figsize=(10,5))
plt.bar(cls_df["Class"],cls_df["Objects"])
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR/"class_distribution.png",dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(obj_per_image,bins=20)
plt.xlabel("Objects per image")
plt.tight_layout()
plt.savefig(FIG_DIR/"objects_per_image.png",dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(bbox_sizes,bins=30)
plt.xlabel("Normalized bounding box area")
plt.tight_layout()
plt.savefig(FIG_DIR/"bbox_size_distribution.png",dpi=300)
plt.close()

print(summary_df)
print("\\nValidation complete.")
