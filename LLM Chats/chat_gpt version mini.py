# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:54:22 2025

@author: myokk
"""

import time
import pickle
from pathlib import Path

#---Update the name of data_import.py file; comment the one not in use---
#from data_import import load_cod_data
from data_import_testdata import load_cod_data

from prompt_writing_functions import split_vector, glue_to_json, write_initial_prompt_v3 #from prompt_writing_functions.py file
from other_functions import json_list_to_df # from other_functions.py file

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()     # uses OPENAI_API_KEY from .env

# set max chunk size... this might need refining based on what API seems to accept
max_chunk_size = 15
sleep_time_between_chunks = 0
model = "gpt-4o-mini-2024-07-18"
output_name = "output_openai_gpt_4o_mini"

# the options are taken from the following paper...
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3229033/#:~:text=Current%20smokers%20had%20significantly%20higher,23.93)%2C%20smoking%2Drelated%20cancers
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
list_x = list(split_vector(cod_vector, max_chunk_size))
prompts_list = [glue_to_json(chunk) for chunk in list_x]
vectors = range(len(prompts_list))

# create an output list same length
llm_output = [None] * len(prompts_list)

# output RDS equivalent
output_path = Path("./Data") / f"{output_name}.pkl"

# -------------------------------------------------------------------
# Loop through chunks and call OpenAI 
# -------------------------------------------------------------------
for i in vectors:

    system_prompt = write_initial_prompt_v3(options)

    # Call OpenAI model
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts_list[i]},
        ],
    )

    # Extract response text
    llm_output[i] = resp.choices[0].message.content

    print(f"completed {i+1} of {len(vectors)}")

    # Save intermediate results
    with open(output_path, "wb") as f:
        pickle.dump(llm_output, f)

    # same as Sys.sleep()
    time.sleep(sleep_time_between_chunks)

# -------------------------------------------------------------------
# The final output as a single DataFrame:
# -------------------------------------------------------------------

df = json_list_to_df(llm_output)
print("Final DataFrame created!")
print(df.head())
