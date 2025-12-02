# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:49:46 2025

@author: myokk
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

#%% Paths and config

BASE_DIR = Path().resolve()
DATA_DIR = BASE_DIR / "Data"

INPUT_CSV = DATA_DIR / "cod_human_labels.csv"   # full dataset (3579 rows)
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VALID_JSONL = DATA_DIR / "valid.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"  # for later evaluation

RANDOM_SEED = 42
VALID_FRAC = 0.2  # 20% of balanced data for validation
TEST_FRAC = 0.2   # 20% of balanced data for test

# Options (same as in the original model)
options = [
    "ischaemic heart disease",
    "cerebrovascular disease",
    "pulmonary disease",
    "lung cancer",
    "colorectal cancer",
    "larynx cancer",
    "kidney cancer",
    "acute myeloid leukemia",
    "oral cavity cancer",
    "esophageal cancer",
    "pancreatic cancer",
    "bladder cancer",
    "stomach cancer",
    "prostate cancer",
    "none",
]
options_str = ", ".join(f'"{o}"' for o in options)

SYSTEM_PROMPT = f"""
You are an actuarial assistant specialising in cause-of-death classification.
Classify the underlying cause of death into ONE of the following categories:
{options_str}.

Return ONLY the category text, with no explanation.

If a cause of death cannot be linked to smoking in any way, for example if it is an infectious disease,
a genetic disorder, or has an external cause provided in the cause_of_death text (e.g. asbestos),
then assign the category as "none".
"""

#%% Load and clean full dataset
print(f"Loading labelled COD data from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Normalise text
df["cause_of_death"] = (
    df["cause_of_death"].astype(str).str.lower().str.strip()
)
df["category_human"] = (
    df["category_human"].astype(str).str.lower().str.strip()
)

# Sanity check: fill any missing labels as "none"
df["category_human"] = df["category_human"].fillna("none")

print("\nFull dataset class distribution:")
print(df["category_human"].value_counts().to_string())

#%% Drop classes with only 1 sample from the fine-tuning set to allow stratification
vc = df["category_human"].value_counts()
rare_classes = vc[vc < 2].index

if len(rare_classes) > 0:
    print("Dropping rare classes with <2 samples from training:", list(rare_classes))
    new_df = df[~df["category_human"].isin(rare_classes)]

print("\nNew dataset class distribution (for fine-tuning, after dropping rare):")
print(new_df["category_human"].value_counts().to_string())
print("Total rows:", len(new_df))

#%% Stratified Split: 80% Train+Val vs 20% Test
trainval_df, test_df = train_test_split(
    new_df,
    test_size = TEST_FRAC,          # 20% held-out test set
    stratify = new_df["category_human"],    # preserve class distribution
    random_state = RANDOM_SEED
)

print("Train/Val size:", len(trainval_df))
print("Test size:", len(test_df))

print("\nTrain/Val label distribution:")
print(trainval_df["category_human"].value_counts(normalize=True))

print("\nTest label distribution:")
print(test_df["category_human"].value_counts(normalize=True))

#%% Split TrainVal Into 75% Train and 25% Validation
train_df, val_df = train_test_split(
    trainval_df,
    test_size = VALID_FRAC * (1 - TEST_FRAC),           # 25% of the 80% TrainVal = 20% overall
    stratify = trainval_df["category_human"],
    random_state = RANDOM_SEED
)

print("Train size:", len(train_df))   # ~60% of full dataset
print("Val size:", len(val_df))       # ~20% of full dataset

# Verify All Splits Are Balanced
print("\nTraining label distribution:")
print(train_df["category_human"].value_counts(normalize=True))

print("\nValidation label distribution:")
print(val_df["category_human"].value_counts(normalize=True))

print("\nTest label distribution:")
print(test_df["category_human"].value_counts(normalize=True))

#%% Build JSONL records for OpenAI fine-tuning
def make_record(cod_text: str, label: str):
    """One training example in OpenAI chat fine-tune format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Cause of death: {cod_text}"},
            {"role": "assistant", "content": label},
        ]
    }


def write_jsonl(df_in: pd.DataFrame, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df_in.iterrows():
            rec = make_record(row["cause_of_death"], row["category_human"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


write_jsonl(train_df, TRAIN_JSONL)
write_jsonl(val_df, VALID_JSONL)
write_jsonl(test_df, TEST_JSONL)

test_df.to_csv(DATA_DIR / 'test_data.csv', index=False)

print(f"\nWrote training data to:   {TRAIN_JSONL}")
print(f"Wrote validation data to: {VALID_JSONL}")
print(f"Wrote test data to: {TEST_JSONL}")
print("\nDone. You can now run ft_cod_openai.py to start fine-tuning.")

