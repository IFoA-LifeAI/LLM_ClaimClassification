# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 11:13:15 2025

@author: myokk
"""

#import re
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(1)

#DATA_FP = Path("./Data/34506561512084822.csv")
DATA_FP = Path("./Data/test_data.csv")

def _clean_names(columns):
    """Rough equivalent of janitor::clean_names()."""
    cols = (
        pd.Series(columns)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.replace(r"(^_|_$)", "", regex=True)
    )
    return cols.tolist()


def load_cod_data():
    """
    Load the cause-of-death data and return:
      cod_data: DataFrame with clean cause_of_death
      cod_vector: shuffled list of unique cause_of_death strings
    """
    
    # read csv like readr::read_csv(..., skip = 10)
    # data = (
    #     pd.read_csv(DATA_FP, skiprows=10) 
    #     .iloc[1:8225]  # R: data[2:8225,]
    #     .copy()
    # )
    cod_data = pd.read_csv(DATA_FP)
    cod_data.columns = _clean_names(cod_data.columns)

    # sum across all numeric columns except 'cause_of_death'
    # cols_to_sum = [c for c in data.columns if c != "cause_of_death"]
    # data[cols_to_sum] = data[cols_to_sum].apply(pd.to_numeric, errors="coerce")
    # data["total"] = data[cols_to_sum].sum(axis=1)

    # cod_data = 
    # cod_data = (
    #     data[["cause_of_death", "total"]]
    #     .rename(columns={"cause_of_death": "cause_of_death_full"})
    #     .query("total > 0")
    #     .copy()
    # )

    # cod_data["letter"] = cod_data["cause_of_death_full"].str.extract(
    #     r"^([A-Z])", expand=False
    # )
    # cod_data["big_number"] = (
    #     cod_data["cause_of_death_full"]
    #     .str.extract(r"^[A-Z](\d{2})", expand=False)
    #     .astype(float)
    # )
    # cod_data["sub_number"] = (
    #     cod_data["cause_of_death_full"]
    #     .str.extract(r"^[A-Z](\d{2}\.\d)", expand=False)
    #     .astype(float)
    # )

    cod_data["cause_of_death"] = (
         cod_data["cause_of_death"]
    #     .str.slice(6)
         .str.replace(r"[)(:,]", "", regex=True)
         .str.strip()
     )

    # cod_data["cause_of_death"] = (
    #     cod_data["cause_of_death_full"]
    #     .str.slice(6)
    #     .str.replace(r"[)(:,]", "", regex=True)
    #     .str.strip()
    # )

    # unique & shuffled cause_of_death vector
    cod_vector = (
        cod_data["cause_of_death"]
        .dropna()
        .drop_duplicates()
        .sample(frac=1, random_state=1)  # shuffle like sample()
        .tolist()
    )

    return cod_data, cod_vector


