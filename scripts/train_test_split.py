from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


# Load the benchmark
df = pd.read_csv("data/financebench/benchmark.csv")
df["company"] = df["company"].str.upper()

# Train + Validation / Test
dev_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

# Train / Validation
train_df, val_df = train_test_split(
    dev_df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

split_dir = Path("data/financebench/splits")
split_dir.mkdir(parents=True, exist_ok=True)

train_df[["question_id"]].to_csv(
    split_dir / "train.csv",
    index=False,
)

val_df[["question_id"]].to_csv(
    split_dir / "validation.csv",
    index=False,
)

test_df[["question_id"]].to_csv(
    split_dir / "test.csv",
    index=False,
)

import json

split_info = {
    "random_seed": 42,
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "test_size": len(test_df),
    "split_strategy": "random",
}

with open(split_dir / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=4)

