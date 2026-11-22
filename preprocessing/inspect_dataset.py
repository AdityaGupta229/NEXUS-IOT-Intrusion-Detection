import pandas as pd
from pathlib import Path

DATASET_PATH = Path("dataset/raw/CICIoT2023")

files = ["train.csv", "test.csv", "validation.csv"]

for file in files:

    print("\n" + "=" * 60)
    print(f"FILE: {file}")
    print("=" * 60)

    # Automatically locate CSV inside nested folders
    matches = list(DATASET_PATH.rglob(file))

    if not matches:
        print(f"ERROR: {file} not found!")
        continue

    file_path = matches[0]

    print("Found at:", file_path)

    df = pd.read_csv(file_path, nrows=5)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nTotal Columns:", len(df.columns))

    print("\nInspection completed!")