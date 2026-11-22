import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "corrected_pipeline"
GRAPH_DIR = RESULTS_DIR / "graphs"

files = {
    "Centralized": "final_centralized_test_report.csv",
    "IID FedAvg": "final_iid_test_report.csv",
    "Non-IID FedAvg": "final_noniid_test_report.csv"
}

reports = {}

for name, filename in files.items():
    df = pd.read_csv(RESULTS_DIR / filename, index_col=0)
    reports[name] = df.iloc[:34]["f1-score"]

comparison = pd.DataFrame(reports).fillna(0)

plt.figure(figsize=(20, 9))

x = np.arange(len(comparison.index))
width = 0.25

for i, model in enumerate(comparison.columns):
    plt.bar(
        x + (i - 1) * width,
        comparison[model],
        width,
        label=model
    )

plt.title("Class-wise F1 Score Comparison")
plt.xlabel("Intrusion Classes")
plt.ylabel("F1 Score")
plt.xticks(x, comparison.index, rotation=90, fontsize=7)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

output = GRAPH_DIR / "classwise_f1_comparison.png"
plt.savefig(output, dpi=250)
plt.close()

print("Saved:", output.name)
print("Class-wise F1 graph generated successfully!")