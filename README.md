# Automated Reef Fish Detection, Counting, and Community Composition Analysis from Underwater Imagery Using YOLOv8

### introduction
This study develops and evaluates a YOLOv8-based object detection framework for automatically detecting, identifying, and quantifying reef fish species from underwater imagery, demonstrating how computer vision can support marine ecological monitoring by reducing manual observation effort while providing quantitative information on fish community composition.

### **What this notebook does, in one sentence:** 
it teaches a computer to find and name 13 different species of fish in photographs, then measures how well it learned and what the fish community in those photos looks like.

### which notebook to run?
Inside the "*.\notebook*", there are two notebooks, identical to the code. I'd recommend to run the **fish_detection.ipynb** code. This is more polished and fine tuned than "ForestIT_SoSe2026.ipynb" file. How to use this notebook is described in the following block

## How to use this notebook (considering you'll do it in google colab)

1. Go to google colab first and ensure you're logged in
2. **Turn the GPU on first:** `Runtime` → `Change runtime type` → `T4 GPU` → `Save`.
<img width="1904" height="872" alt="image" src="https://github.com/user-attachments/assets/e80bd95d-290d-46f3-842e-a70e11fab033" />

3. keep running cell by cell.
-  you might have give permission to access the drive, which is completely okay to do so as it will read and save files in drive.
4. Every phase ends by printing an **EXPECTED OUTPUTS** box that checks the files it was supposed
   to create and marks each one ✓ or ✗. If you see a ✗, stop and read the message.
5. Everything is saved to your Google Drive under **`MyDrive/FishProjectOutputs/`**, so nothing is
   lost when the session ends.

For the sake of easier understanding, I've divided the notebook into 10phases, each phase is burdened with a purpose for your troubleshooting and debugging help, if you're stuck somewhere, it'd be easier to track back and check the checkpoint. 

## The 10 phases

| SL | Phase | What it produces 
|---|---|---|
| 1 | Install libraries | Packages installed, versions printed |
| 2 | Drive, GPU check, settings | Drive mounted, GPU confirmed, every path and setting defined | 
| 3 | Get the dataset | Dataset downloaded from Kaggle, cached in Drive, copied to fast local disk |
| 4 | Dataset validation + exploration | 3 CSVs, a text report, 4 figures, sample images | 3–8 min |
| 5 | **YOLOv8 training** | `best.pt`, `last.pt`, `results.csv`, training curves — **checkpointed every epoch, resumable** | 
| 5R | Recovery if disconnected somehow | Puts the session back together without retraining | 
| 6 | Evaluation | Confusion matrix, PR curve, F1 curve, precision/recall/mAP, per-class table | 
| 7 | Fish counting + community composition | Counting CSVs, annotated images, abundance table, diversity indices, 3 figures | 
| 8 | Confidence threshold analysis | Threshold comparison table and plots | 
| 9 | Error analysis | False positives, false negatives, annotated examples, summary | 

**Phase 5R is not part of the normal run.** It is only required if Colab is somehow disconnected. If the session is still healthy it detects that and skips itself, so it is safe to leave in a "Run all" command too.

## If Colab disconnects

It is very much possible that several times your COLAB might get disconnected, maybe due to inactivity or due to internet disruption. However, don't worry, this notebook ensures that Nothing is lost after the training phase. Once the training is saved, then a fallback checkpoint is to your Drive **after every single epoch**.

- **Disconnected during Phase 5 (training)?**

  If disconnected before training the dataset, then Re-run Phase 1, Phase 2, then Phase 5 again. It   finds the checkpoint in your Drive and carries on from the epoch it reached. It does not start over.
- **Disconnected after training finished?** 

Re-run Phase 1, Phase 2, then **Phase 5R**, then jump straight to whichever phase you were on.

## Where the data comes from

Kaggle: <https://www.kaggle.com/datasets/mahmoodyousaf/fish-dataset> — "Fish Dataset", already in YOLO format, 363 MB, Apache 2.0 licence, originally sourced via Roboflow. 13 fish species. Phase 3 downloads it.

Based on this dataset, this academic project is done for the course "**Innovations and Applications of Forest IT**". All the outputs are saved here and the report will be added once it's graded. 
