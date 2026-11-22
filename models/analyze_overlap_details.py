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

feature_cols = [
    col for col in train_df.columns
    if col != "label"
]

train_hashes = set(
    pd.util.hash_pandas_object(
        train_df,
        index=False
    ).tolist()
)

print("Checking overlapping test records...")

overlap_rows = []

for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE):

    hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    )

    mask = hashes.isin(train_hashes)

    if mask.any():
        overlap_rows.append(chunk.loc[mask])

overlap_df = pd.concat(
    overlap_rows,
    ignore_index=True
)

print("\nTotal overlapping rows:", len(overlap_df))

print("\nOverlapping records by class:")

print(
    overlap_df["label"]
    .value_counts()
    .to_string()
)

# Check identical features with different labels

train_feature_hashes = set(
    pd.util.hash_pandas_object(
        train_df[feature_cols],
        index=False
    ).tolist()
)

test_feature_hashes = pd.util.hash_pandas_object(
    overlap_df[feature_cols],
    index=False
)

train_label_map = {}

for feature_hash, label in zip(
    pd.util.hash_pandas_object(
        train_df[feature_cols],
        index=False
    ),
    train_df["label"]
):
    train_label_map.setdefault(
        feature_hash,
        set()
    ).add(label)

conflicts = 0

for feature_hash, test_label in zip(
    test_feature_hashes,
    overlap_df["label"]
):
    if test_label not in train_label_map.get(
        feature_hash, set()
    ):
        conflicts += 1

print("\nFeature-label conflict count:", conflicts)

print("\nOverlap analysis completed!")