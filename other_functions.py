# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 12:27:57 2025

@author: myokk
"""

import json
import re
import time
from typing import List, Optional

import pandas as pd

from prompt_writing_functions import split_vector, glue_to_json, write_initial_prompt_v3

# Load env vars (for OPENAI_API_KEY from .env)
from dotenv import load_dotenv
load_dotenv()

# Create OpenAI client (uses OPENAI_API_KEY env var)
from openai import OpenAI
client = OpenAI()


def json_list_to_df(output_list: List[str]) -> pd.DataFrame:
    """
     - Takes a list of raw LLM string responses
    - If there are ``` code fences, uses the LAST fenced block
    - Strips backticks and leading 'json'
    - Parses JSON into Python objects
    - Combines into a single pandas DataFrame
    """
    rows = []
    # i = 1
    for x in output_list:
        # print(i)
        # Check for ``` fenced blocks
        if "```" in x:
            matches = re.findall(r"```(.*?)```", x, flags=re.S)
            if matches:
                # Use the last fenced block 
                x = matches[-1]

        # Remove backticks and leading 'json'
        x_clean = x.replace("`", "")
        x_clean = re.sub(r"^\s*json", "", x_clean, flags=re.I).strip()
        
        x_clean = re.sub(r",\s*}", "}", x_clean)
        x_clean = re.sub(r",\s*]", "]", x_clean)
        if not x_clean.endswith("]") and x_clean.strip().startswith("["):
            x_clean += "]"
        if not x_clean.endswith("}") and x_clean.strip().startswith("{"):
            x_clean += "}"
        
        try:
            parsed = json.loads(x_clean)
            if isinstance(parsed, dict):
                rows.append(parsed)
            elif isinstance(parsed, list):
                rows.extend(parsed)
            else:
                raise ValueError("Parsed JSON is neither a dict nor a list of dicts.")

        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            print("Problematic string:\n", x_clean)
            continue

        # i = i + 1
    return pd.DataFrame(rows)

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

def classify_data_with_llm(
    data: List[str],
    max_chunk_size: int = 15,
    sleep_time_between_chunks: float = 0,
    model: str = "gpt-4o-2024-08-06",
    message_updates: bool = True,
    initial_prompt: Optional[str] = write_initial_prompt_v3(options),
) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : list of str
        The data to be classified (e.g. list of cause_of_death strings).
    max_chunk_size : int
        Max number of items per chunk sent to the LLM.
    sleep_time_between_chunks : float
        Seconds to sleep between API calls.
    model : str
        OpenAI model name.
    message_updates : bool
        If True, prints progress messages.
    initial_prompt : str, optional
        System prompt to send to the LLM.
        If None, raises an error (you should call write_initial_prompt_v3(options) and pass it in).

    Returns
    -------
    pandas.DataFrame
        Combined JSON responses from all chunks, as a DataFrame.
    """

    if initial_prompt is None:
        raise ValueError(
            "initial_prompt must be provided (e.g. write_initial_prompt_v3(options))."
        )

    # Chunk the data and turn each chunk into a JSON string
    list_x = list(split_vector(data, max_chunk_size))
    prompts_list = [glue_to_json(chunk) for chunk in list_x]
    vectors = range(len(prompts_list))

    # Output container (one response per chunk)
    llm_output: List[Optional[str]] = [None] * len(prompts_list)

    # Loop through chunks and call the LLM
    for i in vectors:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": prompts_list[i]},
            ],
        )

        # Store raw text response
        llm_output[i] = resp.choices[0].message.content

        if message_updates:
            print(f"completed {i + 1} of {len(vectors)}")

        if sleep_time_between_chunks > 0:
            time.sleep(sleep_time_between_chunks)

    # Clean + combine into DataFrame
    output_df = json_list_to_df(llm_output)
    return output_df
