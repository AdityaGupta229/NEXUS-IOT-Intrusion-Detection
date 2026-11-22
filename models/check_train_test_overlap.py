import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "dataset" / "processed" / "train_sample.csv"

TEST_PATH = (
    BASE_DIR / "dataset" / "raw" / "CICIoT2023" /
    "CICIOT23" / "test" / "test.csv"
)

CHUNK_SIZE = 20000

print("Loading training sample...")

train_df = pd.read_csv(TRAIN_PATH)

columns = train_df.columns.tolist()

train_hashes = set(
    pd.util.hash_pandas_object(
        train_df[columns],
        index=False
    ).tolist()
)

print("Training records:", len(train_df))
print("Checking test dataset for exact duplicates...")

overlap_count = 0
total_records = 0

for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE):

    chunk = chunk[columns]

    hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    )

    overlap_count += hashes.isin(train_hashes).sum()
    total_records += len(chunk)

    print(
        f"Checked {total_records} records | "
        f"Overlaps: {overlap_count}"
    )

print("\n========== OVERLAP AUDIT ==========")
print("Training sample records:", len(train_df))
print("Test records:", total_records)
print("Exact duplicate records:", overlap_count)

print(
    "Overlap percentage:",
    round((overlap_count / total_records) * 100, 4),
    "%"
)

print("\nAudit completed!")