import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("dataset/raw/CICIoT2023")
CHUNK_SIZE = 100_000

files = ["train.csv", "test.csv", "validation.csv"]

for filename in files:

    matches = list(DATASET_PATH.rglob(filename))

    if not matches:
        print(f"File not found: {filename}")
        continue

    file_path = matches[0]

    print("\n" + "=" * 60)
    print("ANALYZING:", filename)
    print("=" * 60)

    total_rows = 0
    missing_values = Counter()
    label_counts = Counter()
    infinite_values = Counter()

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        # Count missing values
        missing_values.update(chunk.isnull().sum().to_dict())

        # Count class labels
        label_counts.update(
            chunk["label"].value_counts().to_dict()
        )

        # Count infinite numeric values
        numeric = chunk.select_dtypes(include=np.number)

        inf_counts = np.isinf(numeric).sum()
        infinite_values.update(inf_counts.to_dict())

    print("\nTotal Records:", total_rows)

    print("\nMissing Values:")
    print({k: v for k, v in missing_values.items() if v > 0}
          or "No missing values")

    print("\nInfinite Values:")
    print({k: v for k, v in infinite_values.items() if v > 0}
          or "No infinite values")

    print("\nTotal Unique Classes:", len(label_counts))

    print("\nClass Distribution:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count:,}")

print("\nDATASET ANALYSIS COMPLETED!")