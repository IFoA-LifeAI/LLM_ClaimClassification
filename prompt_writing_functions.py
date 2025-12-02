# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:48:11 2025

@author: myokk
"""

from typing import List
import json

def _format_options_lines(options: List[str]) -> str:
    """
    Turn a list like ["a", "b"] into:
    "a"
    "b"
    (each option on its own line, wrapped in double quotes)
    """
    return "\n".join(f"\"{opt}\"" for opt in options)


def write_initial_prompt(options: List[str]) -> str:
    """
    Python equivalent of write_intial_prompt() in R.
    """
    options_block = _format_options_lines(options)
    return (
        "You are a classification LLM. You will receive a JSON file. "
        "The file will contain a list of items with cause_of_death.\n"
        "You must only return the edited version of this JSON file. "
        "Please add 'category' to each item, which can only ever have one of the following values:\n"
        f"{options_block}\n\n"
        "No capitalization. No explanations. Return only the data in a structured JSON format."
    )


def write_initial_prompt_v2(options: List[str]) -> str:
    """
    Python equivalent of write_initial_prompt_v2() in R.
    """
    options_block = _format_options_lines(options)
    return (
        "You are a classification LLM. You will receive a JSON file. "
        "The file will contain a list of items with cause_of_death.\n"
        "It is important that you return only an edited version of the JSON file. "
        "Add 'category' to each item, which can only ever pick one of the values below. "
        'If none are suitable choose the category of "none":\n\n'
        f"{options_block}\n\n"
        "No explanations. Return only the data in a structured JSON format. "
        "Your final JSON code must begin with ``` and end with ```"
    )


def write_initial_prompt_v3(options: List[str]) -> str:
    """
    Python equivalent of write_initial_prompt_v3() in R.
    """
    options_block = _format_options_lines(options)
    return (
        "You are a classification LLM. You will receive a JSON file. "
        "The file will contain a list of items with cause_of_death.\n"
        "It is important that you return only an edited version of the JSON file. "
        "Add 'category' to each item, which can only ever pick one of the values below. "
        'If none are suitable choose the category of "none":\n\n'
        f"{options_block}\n\n"
        "No explanations. Return only the data in a structured JSON format. "
        "Your final JSON code must begin with ``` and end with ```.\n"
        "If a cause of death cannot be linked to smoking in any way, for example if it is an infectious disease, "
        "a genetic disorder, or has an external cause provided in the cause_of_death text (e.g. asbestos), "
        'then assign the category as "none".'
    )

def write_initial_prompt_v4(options: List[str]) -> str:

    options_block = _format_options_lines(options)
    return (f"""
    You are a classification LLM. You will receive a JSON array. 
    Each element of the array will contain a field named "cod" (cause of death).
    
    Your task:
    - Return a JSON object with a single field:
        "items": [ ... ]
    - The "items" array must contain one object per input COD.
    - Each object must contain one field: "category".
    
    Allowed categories:
    {options_block}
    
    Rules:
    - Choose exactly ONE category per item.
    - If none are suitable, return "none".
    - For example, infectious diseases, genetic disorders, accidents, or external causes (e.g., asbestos) must be classified as "none".
    
    STRICT OUTPUT REQUIREMENTS:
    - Return ONLY valid JSON.
    - No explanations, no markdown, no code fences.
    - No additional text before or after the JSON.
    - Output must strictly match the JSON schema provided by the system.
    """
    )

def glue_chr_vector(vec: List[str]) -> str:
    """
    Python equivalent of glue_chr_vector():
    Convert a list of strings into a single string with each value on its own line, quoted.
    e.g. ["a", "b"] -> "\"a\"\n\"b\""
    """
    return "\n".join(f"\"{v}\"" for v in vec)


def split_vector(vec: List[str], max_length_per_vec: int):
    """
    Python equivalent of split_vector():
    Split a list into chunks of length <= max_length_per_vec.
    Returns a generator of sublists.
    """
    for i in range(0, len(vec), max_length_per_vec):
        yield vec[i: i + max_length_per_vec]


def glue_to_json(vec: List[str]) -> str:
    return json.dumps([{"cause_of_death": v} for v in vec], ensure_ascii=False)

    """
    Python equivalent of glue_to_json():
    Convert a list of strings into a JSON-like string:
    [
      { "cause_of_death": "..." },
      { "cause_of_death": "..." },
      ...
    ]
    This returns a string (not a Python list/dict), matching the R behaviour.
    """
    """
    inner_lines = ",\n".join(
        f'  {{ "cause_of_death": "{v}" }}' for v in vec
    )
    return "[\n" + inner_lines + "\n]"
    """
