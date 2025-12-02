# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:53:42 2025

@author: myokk
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

#%% Config and helpers
# Folder where your .pkl outputs exist
BASE_DIR = Path().resolve()
DATA_DIR = BASE_DIR / "Data"

# Model files (comment out ones you don't use)
model_files = {
    "gpt_4o_mini": "output_openai_gpt_4o_mini.pkl",
    "gpt_4o": "output_openai_gpt_4o.pkl",
    "gpt_4o_mini_ft": "output_openai_gpt_4o_mini_ft.pkl",
    #"gpt_4o_ft": "output_openai_gpt_4o_ft.pkl",
    # "llama33": "output_groq_llama33.pkl",
    # "deepseek_r1": "output_deepseek_r1.pkl",
    # "gemini": "output_gemini.pkl",
}

# the options are taken from the following paper...
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3229033/#:~:text=Current%20smokers%20had%20significantly%20higher,23.93)%2C%20smoking%2Drelated%20cancers
options = [
    "ischaemic heart disease", #https://archive.datadictionary.nhs.uk/DD%20Release%20March%202021/Covid19PRA/Coronary_Heart.html
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

# note CDH is not the same as aortic aneurysm, though both are related
# https://pmc.ncbi.nlm.nih.gov/articles/PMC7711307/#:~:text=Even%20though%20abdominal%20aortic%20aneurysm,but%20also%20several%20important%20differences.

cat_dtype = CategoricalDtype(categories = options, ordered = False)

#%% Import data

#---Update the name of data_import.py file; comment the one not in use---
#from data_import import load_cod_data
from data_import_testdata import load_cod_data

from other_functions import json_list_to_df

# -------------------------------------------------------------------
# 1. Helper: load a single LLM .pkl output and clean it
# -------------------------------------------------------------------

def load_llm_output_pkl(path: Path) -> pd.DataFrame:
    """
    Load .pkl containing a Python list of JSON strings,
    convert to a tidy DataFrame, drop duplicates, lowercase strings.
    """
    with open(path, "rb") as f:
        llm_list = pickle.load(f)

    df = json_list_to_df(llm_list)
    df = df.drop_duplicates()

    # lower-case all string columns
    df = df.apply(
        lambda col: col.str.lower() if col.dtype == "object" else col
    )
    return df

# -------------------------------------------------------------------
# 2. Load cod_data and human labels
# -------------------------------------------------------------------

cod_data, cod_vector = load_cod_data()

# Adjust this path if your Excel lives elsewhere
human_fp = DATA_DIR / "Human Classification v2.xlsx"
output_human = pd.read_excel(human_fp, sheet_name="category_human")

# Make sure human labels are lower case
output_human = output_human.copy()
output_human["cause_of_death"] = output_human["cause_of_death"].str.lower()
output_human["category_human"] = output_human["category_human"].str.lower()

# -------------------------------------------------------------------
# 3. Load all available LLM model outputs
# -------------------------------------------------------------------

model_dfs = {}  # {model_name: DataFrame}

for model_name, filename in model_files.items():
    path = DATA_DIR / filename
    if path.exists():
        print(f"Loading {model_name} from {path} ...")
        df_model = load_llm_output_pkl(path)

        # We expect df_model to have at least ['cause_of_death', 'category']
        if "cause_of_death" not in df_model.columns or "category" not in df_model.columns:
            raise ValueError(f"{filename} must contain 'cause_of_death' and 'category' columns.")

        # Rename 'category' -> model-specific name, e.g. 'category_gpt_4o_mini'
        df_model = df_model.rename(columns={"category": f"category_{model_name}"})
        model_dfs[model_name] = df_model[["cause_of_death", f"category_{model_name}"]]
    else:
        print(f"Skipping {model_name}: file not found at {path}")

if not model_dfs:
    raise RuntimeError("No LLM outputs found. Check MODEL_FILES and DATA_DIR paths.")

#%% Combine LLM data imported
# Base DataFrame from cod_vector (like tibble(cause_of_death = cod_vector, category = NA))
results_df = pd.DataFrame({
    "cause_of_death": pd.Series(cod_vector, dtype="string"),
})

# Lowercase cause_of_death to match everything
results_df["cause_of_death"] = results_df["cause_of_death"].str.lower()
results_df = results_df.drop_duplicates()

# Join each model's category column (left join by cause_of_death)
for model_name, df_model in model_dfs.items():
    results_df = results_df.merge(
        df_model,
        on="cause_of_death",
        how="left",
        suffixes=("", f"_{model_name}"),  # mostly irrelevant because we renamed col
    )

# Make all category_* columns categorical with 'options' levels
category_cols = [c for c in results_df.columns if c.startswith("category_")]
for col in category_cols:
    results_df[col] = results_df[col].astype("string")
    results_df[col] = results_df[col].str.lower()
    results_df[col] = results_df[col].astype(cat_dtype)

print("\nresults_df preview (with model categories):")
print(results_df.head())

#%% Determine consensus
def compute_consensus(results_df: pd.DataFrame) -> pd.DataFrame:
    # Use only model category columns (exclude human/consensus columns if present)
    model_category_cols = [
        c for c in results_df.columns
        if c.startswith("category_")
        and not (
            c == "category_human"
            or c == "category_consensus"
            or c == "category_unanimous_consensus"
        )
    ]

    # Long format: one row per (cause_of_death, model, category)
    long_df = results_df.melt(
        id_vars=["cause_of_death"],
        value_vars=model_category_cols,
        var_name="llm_type",
        value_name="category",
    )

    # Drop rows with missing category
    long_df = long_df.dropna(subset=["category"])

    # Group by cause_of_death + category and count how many models chose that category
    grouped = (
        long_df
        .groupby(["cause_of_death", "category"], observed=False)  # silences the FutureWarning
        .agg(
            llm_types=("llm_type", lambda s: list(s)),
            consensus_no=("llm_type", "size"),
        )
        .reset_index()
    )

    # For each cause_of_death, keep rows with the max consensus_no
    grouped = grouped.sort_values(
        ["cause_of_death", "consensus_no"], ascending=[True, False]
    )

    max_per_cod = grouped.groupby("cause_of_death")["consensus_no"].transform("max")
    grouped = grouped[max_per_cod == grouped["consensus_no"]].reset_index(drop=True)

    # Unanimous consensus = all models agreed
    n_models = len(model_category_cols)
    grouped["category_unanimous_consensus"] = np.where(
        grouped["consensus_no"] == n_models,
        grouped["category"],
        pd.NA,
    )

    # Identify causes with >1 tied top category (no strict consensus)
    dup_mask = grouped.groupby("cause_of_death").cumcount() > 0
    cause_without_consensus = grouped.loc[dup_mask, "cause_of_death"].unique()

    # Keep only the first row per cause_of_death
    grouped_single = (
        grouped
        .sort_values(["cause_of_death", "consensus_no"], ascending=[True, False])
        .groupby("cause_of_death", as_index=False)
        .head(1)
    )

    grouped_single["without_consensus"] = grouped_single["cause_of_death"].isin(
        cause_without_consensus
    )

    grouped_single = grouped_single[[
        "cause_of_death",
        "category",                    # -> category_consensus
        "consensus_no",
        "without_consensus",
        "category_unanimous_consensus",
    ]].rename(columns={"category": "category_consensus"})

    return grouped_single


results_consensus = compute_consensus(results_df)
print("\nConsensus preview:")
print(results_consensus.head())

#%% Attach human and consensus to results
# Merge consensus (one-to-one by cause_of_death)
results_df = results_df.merge(
    results_consensus,
    on="cause_of_death",
    how="left",
)

# Merge human classifications
results_df = results_df.merge(
    output_human,
    on="cause_of_death",
    how="left",
)

# Clean human labels
results_df["category_human"] = results_df["category_human"].fillna("none").str.lower()
results_df["category_human"] = results_df["category_human"].astype(cat_dtype)

print("\nresults_df with consensus + human preview:")
print(results_df.head())

# Update category_cols to include new category_* columns if needed
category_cols = [c for c in results_df.columns if c.startswith("category_")]

#%% Accuracy and proportion answered
accuracy_records = []

for col in category_cols:
    model_name = col  # full column name, e.g., 'category_gpt_4o_mini'

    model_vals = results_df[col]
    human_vals = results_df["category_human"]

    # Where model is not missing
    valid_mask = model_vals.notna()
    if valid_mask.sum() == 0:
        accuracy = np.nan
    else:
        accuracy = (model_vals[valid_mask] == human_vals[valid_mask]).mean()

    # Proportion answered: size of non-missing model vs non-missing human
    human_notna = human_vals.notna().sum()
    if human_notna == 0:
        proportion_answered = np.nan
    else:
        proportion_answered = model_vals.notna().sum() / human_notna

    accuracy_records.append({
        "model": model_name,
        "accuracy": accuracy,
        "proportion_answered": proportion_answered,
    })

accuracy_df = pd.DataFrame(accuracy_records)

# Make model names nicer
accuracy_df["model"] = accuracy_df["model"].str.replace(r"^category_", "", regex=True)

# Drop 'human' if present
accuracy_df = accuracy_df[accuracy_df["model"] != "human"]

# Sort by accuracy
accuracy_df = accuracy_df.sort_values("accuracy", ascending=False)

accuracy_df["accuracy_pct"] = (accuracy_df["accuracy"] * 100).round(1)
accuracy_df["proportion_answered_pct"] = (accuracy_df["proportion_answered"] * 100).round(0)

print("\nAccuracy and proportion answered by model:")
print(accuracy_df[["model", "proportion_answered_pct", "accuracy_pct"]])

#%% Precision, recall, F1 
# Melt to long format for all model columns except human
model_cols_no_human = [c for c in category_cols if c != "category_human"]

long_df = results_df.melt(
    id_vars=["cause_of_death", "category_human"],
    value_vars=model_cols_no_human,
    var_name="model",
    value_name="category",
)

long_df = long_df.dropna(subset=["category"])
long_df["correct"] = (long_df["category"] == long_df["category_human"])

def confusion_counts(sub: pd.DataFrame) -> pd.Series:
    human = sub["category_human"]
    correct = sub["correct"]

    tp = ((correct) & (human != "none")).sum()
    tn = ((correct) & (human == "none")).sum()
    fp = ((~correct) & (human == "none")).sum()
    fn = ((~correct) & (human != "none")).sum()

    return pd.Series({
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    })

results_stats = (
    long_df
    .groupby("model", as_index=False)
    .apply(confusion_counts, include_groups=False)
)

results_stats["precision"] = (
    results_stats["true_positive"]
    / (results_stats["true_positive"] + results_stats["false_positive"])
    * 100
)
results_stats["recall"] = (
    results_stats["true_positive"]
    / (results_stats["true_positive"] + results_stats["false_negative"])
    * 100
)
results_stats["f1"] = (
    2 * results_stats["precision"] * results_stats["recall"]
    / (results_stats["precision"] + results_stats["recall"])
)

results_stats["model"] = results_stats["model"].str.replace(r"^category_", "", regex=True)
results_stats = results_stats.sort_values("f1", ascending=False)

print("\nPrecision / Recall / F1 by model (including consensus):")
print(results_stats[[
    "model",
    "precision", "recall", "f1",
    "true_positive", "true_negative",
    "false_positive", "false_negative",
]])

results_stats_no_consensus = results_stats[~results_stats["model"].str.contains("consensus")]
print("\nNon-consensus models only:")
print(results_stats_no_consensus[["model", "precision", "recall", "f1"]])

results_stats_consensus = results_stats[results_stats["model"].str.contains("consensus")]
print("\nConsensus models only:")
print(results_stats_consensus[["model", "precision", "recall", "f1"]])

#%% “Messing around” views (disagreements)

# 1) Use only true model columns, not human/consensus
category_cols = [
    c for c in results_df.columns
    if c.startswith("category_")
    and c not in ["category_human", "category_consensus", "category_unanimous_consensus"]
]

# 2) Make string copies with NA replaced by a sentinel
cats_str = results_df[category_cols].apply(
    lambda s: s.astype("object").where(~s.isna(), "__NA__")
)
human_str = results_df["category_human"].astype("object").where(
    ~results_df["category_human"].isna(), "__NA__"
)

cats = cats_str.to_numpy(dtype="object")                 # shape (n_rows, n_models)
human = human_str.to_numpy(dtype="object")              # shape (n_rows,)

# 3) Compare model prediction vs human, elementwise
cmp = cats != human[:, None]

# 4) Ignore places where either side was originally NA
valid = (~cats_str.isna()).to_numpy(dtype=bool) & (~results_df["category_human"].isna().to_numpy(dtype=bool))[:, None]

diff = cmp & valid

# At least one model disagrees with human for this row
any_diff_mask = diff.any(axis=1)

# 5) Build a mask where consensus is non-missing AND != human
cons = results_df["category_unanimous_consensus"].astype("object")
hum = results_df["category_human"].astype("object")

cons_not_na = ~results_df["category_unanimous_consensus"].isna()
cons_diff_human = cons != hum

mask_cons = cons_not_na & cons_diff_human

# 6) Build mismatch_df
mismatch_df = results_df[
    any_diff_mask & mask_cons
][["cause_of_death", "category_human", "category_unanimous_consensus"]].copy()

mismatch_df = mismatch_df.sort_values(
    ["category_human", "category_unanimous_consensus"]
)

print("\nCases where unanimous consensus != human (and some model disagrees):")
print(mismatch_df.head())

# 7) Rows where some model predicted "none" but human != "none"
def any_model_none_vs_non_none(row) -> bool:
    human = row["category_human"]
    if human == "none":
        return False
    return any(
        (row[col] == "none") and (row[col] != human)
        for col in category_cols
    )

mask_none_vs_non_none = results_df.apply(any_model_none_vs_non_none, axis=1)
df_none_vs_non_none = results_df[mask_none_vs_non_none]

print("\nRows where at least one model predicted 'none' but human != 'none':")
print(df_none_vs_non_none.head())

print("\nDone.")

# Save .csv file to use as INPUT_CSV for model fine-tuning; if you don't need, comment out.
csv_path = DATA_DIR / "cod_human_labels.csv"
results_df[["cause_of_death", "category_human"]].to_csv(csv_path, index=False)
