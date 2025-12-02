# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:19:49 2025

@author: myokk
"""

#import json
import time
import pickle
import pandas as pd
from pathlib import Path
from typing import List

#---Update the name of data_import.py file; comment the one not in use---
#from data_import import load_cod_data
from data_import_testdata import load_cod_data

from prompt_writing_functions import  split_vector, glue_to_json, write_initial_prompt_v3
from other_functions import json_list_to_df

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

BASE_DIR = Path().resolve()
DATA_DIR = BASE_DIR / "Data"

train_file_path = DATA_DIR / "train.jsonl"
valid_file_path = DATA_DIR / "valid.jsonl"

#%% Upload files
train = client.files.create(
    file=open(train_file_path, "rb"),
    purpose="fine-tune"
)
valid = client.files.create(
    file=open(valid_file_path, "rb"),
    purpose="fine-tune"
)

print("Uploaded training file:", train.id)
print("Uploaded validation file:", valid.id)

#%% Fine-tune LLM model
# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-2024-08-06",   # base model
    training_file=train.id,
    validation_file=valid.id,
    hyperparameters={
        "n_epochs": 2,   # recommend 2–4 for the given training dataset
    },
)

print("JOB_ID:", job.id)

# Run fine-tuning job
while True:
    j = client.fine_tuning.jobs.retrieve(job.id)
    print("Status:", j.status)
    if j.status in ("succeeded", "failed", "cancelled"):
        print("Final status:", j.status)
        print("Fine-tuned model:", j.fine_tuned_model)
        break
    time.sleep(30)
    
#%% COD Batch Classifier Setup
# Replace with the model name printed by your fine-tune job
ft_model = "ft:gpt-4o-mini-2024-07-18:personal::Cgtargvn"
output_name = "output_openai_gpt_4o_ft"

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

# -------------------------------------------------------------------
# Load cod_vector 
# -------------------------------------------------------------------

cod_data, cod_vector = load_cod_data()

# -------------------------------------------------------------------
# Prepare chunked prompts 
# -------------------------------------------------------------------
def classify_cod_batch(
    data: List[str],
    max_chunk_size: int = 12,
    sleep_time_between_chunks: float = 0,
    model: str = ft_model,
    message_updates: bool = True,
) -> pd.DataFrame:

    # Split into chunks
    list_x = list(split_vector(data, max_chunk_size))

    # Convert each chunk to a JSON input array for the LLM
    prompts_list = [glue_to_json(chunk) for chunk in list_x]
    vectors = range(len(prompts_list))
    
    # create an output list same length
    llm_output = [None] * len(prompts_list)
    
    # output RDS equivalent
    output_path = DATA_DIR / f"{output_name}.pkl"
                
    system_prompt = write_initial_prompt_v3(options)

    for i in vectors:

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts_list[i]},
            ],
            temperature=0,
        )

        llm_output[i] = resp.choices[0].message.content

        if message_updates:
            print(f"completed {i+1} of {len(vectors)}")
            
        # Save intermediate results 
        with open(output_path, "wb") as f:
            pickle.dump(llm_output, f)

        if sleep_time_between_chunks > 0:
            time.sleep(sleep_time_between_chunks)

    # Parse and combine into DataFrame
    df = json_list_to_df(llm_output)

    return df
       
#%% Run the test dataset
results_df = classify_cod_batch(
    cod_vector,
    max_chunk_size=12,
    model=ft_model
)

print("Final DataFrame created!")
print(results_df.head())
