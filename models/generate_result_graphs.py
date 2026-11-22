import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "corrected_pipeline"
GRAPH_DIR = RESULTS_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load final comparison
df = pd.read_csv(
    RESULTS_DIR / "final_model_comparison.csv"
)

# Convert metrics to percentages
metrics = ["Accuracy", "Macro F1", "Weighted F1"]

for metric in metrics:
    df[metric] = df[metric] * 100

# Create comparison charts
for metric in metrics:

    plt.figure(figsize=(9, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Model")
    plt.ylabel(f"{metric} (%)")
    plt.ylim(0, 100)

    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(df[metric]):
        plt.text(i, value + 1, f"{value:.2f}%", ha="center")

    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + "_comparison.png"

    plt.savefig(GRAPH_DIR / filename, dpi=300)
    plt.close()

    print("Saved:", filename)

print("\nAll comparison graphs generated successfully!")