"""
05_community_composition.py
Ecological summary built from Notebook 04's detection counts:
abundance, relative abundance, dominant species, and diversity indices.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "community"
FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

counts_path = PROJECT_ROOT / "outputs" / "counting" / "species_totals.csv"
species_totals = pd.read_csv(counts_path)

# ----------------------------------------------------------------
# Abundance & relative abundance
# ----------------------------------------------------------------
total_individuals = species_totals["Total_Count"].sum()
species_totals["Relative_Abundance_%"] = (
    species_totals["Total_Count"] / total_individuals * 100
).round(2)

species_totals = species_totals.sort_values("Total_Count", ascending=False).reset_index(drop=True)
species_totals["Rank"] = species_totals.index + 1
species_totals.to_csv(OUTPUT_DIR / "abundance_table.csv", index=False)

# guard against species with zero detections when computing "rarest"
present = species_totals[species_totals["Total_Count"] > 0]
dominant_species = present.iloc[0]["Species"]
rarest_species = present.iloc[-1]["Species"]

# ----------------------------------------------------------------
# Diversity indices (computed only over species actually detected)
# ----------------------------------------------------------------
p = present["Total_Count"] / total_individuals
shannon_index = -np.sum(p * np.log(p))
simpson_index = 1 - np.sum(p ** 2)
species_richness = present.shape[0]
pielou_evenness = shannon_index / np.log(species_richness) if species_richness > 1 else np.nan

diversity_summary = pd.DataFrame({
    "Metric": ["Total Individuals Detected", "Species Richness (detected)",
               "Shannon Index (H')", "Simpson Index (1-D)", "Pielou Evenness (J')",
               "Dominant Species", "Rarest Detected Species"],
    "Value": [total_individuals, species_richness,
              round(shannon_index, 3), round(simpson_index, 3),
              round(pielou_evenness, 3) if pd.notna(pielou_evenness) else "n/a",
              dominant_species, rarest_species]
})
diversity_summary.to_csv(OUTPUT_DIR / "diversity_indices.csv", index=False)
print(diversity_summary)

# ----------------------------------------------------------------
# Figures
# ----------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(species_totals["Species"], species_totals["Total_Count"], color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Abundance (individuals detected)")
plt.title("Species Abundance - Test Set Detections")
plt.tight_layout()
plt.savefig(FIG_DIR / "community_abundance.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 10))
plt.pie(present["Total_Count"], labels=present["Species"], autopct="%1.1f%%", startangle=90)
plt.title("Relative Abundance by Species")
plt.savefig(FIG_DIR / "community_relative_abundance.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(present["Rank"], present["Total_Count"], marker="o")
plt.yscale("log")
plt.xlabel("Species Rank")
plt.ylabel("Abundance (log scale)")
plt.title("Rank-Abundance Curve")
plt.tight_layout()
plt.savefig(FIG_DIR / "rank_abundance_curve.png", dpi=300)
plt.close()

print("Community composition analysis complete.")
