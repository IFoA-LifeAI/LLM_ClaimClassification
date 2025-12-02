# LLM-Based Cause of Death Classification
A Python pipeline for automated cause-of-death (COD) classification using Large Language Models (LLMs).
The project compares zero-shot and fine-tuned models against human, consensus, and unanimous consensus labels.


**Objectives**
- Classify COD descriptions into 15 smoking-related categories
- Compare zero-shot vs. fine-tuned LLM performance
- Evaluate accuracy, precision, recall, and F1 metrics


**Cause of Death Categories**
- Ischaemic heart disease
- Cerebrovascular disease
- Pulmonary disease
- Lung cancer
- Colorectal cancer
- Larynx cancer
- Kidney cancer
- Acute myeloid leukemia
- Oral cavity cancer
- Esophageal cancer
- Pancreatic cancer
- Bladder cancer
- Stomach cancer
- Prostate cancer
- None


**Pipeline Overview**

**1. Zero-Shot Classification**

Generates predictions using:
- gpt-4o-mini; run via **LLM Chats/chat_gpt version mini.py**
- gpt-4o; run via **LLM Chats/chat_gpt version.py**
- Human labels, consensus, unanimous consensus

Run: **main.py**

**Note: Update output filenames and data import paths inside .py files before running.**


**2. Fine-Tuning Pipeline**

2.1 Prepare Training Data

Stratified 60/20/20 train/validation/test split.
Run: **python prepare_openai_finetune.py**

2.2 Fine-Tune the Model

Run:

**ft_cod_openai_mini.py**
or
**ft_cod_openai.py**

2.3 Re-Run Classification on Test Set

After fine-tuning:

Generates predictions using:
- gpt-4o-mini; run via **LLM Chats/chat_gpt version mini.py**
- gpt-4o; run via **LLM Chats/chat_gpt version.py**
- ft_gpt-4o-mini; run via **LLM chats/ft_chat_gpt_mini.py**

Run: **main.py**

**Note: Update output filenames and data import paths inside .py files before running.**

Produces:

Zero-shot predictions

Fine-tuned model predictions

Consensus & human comparisons


**3. Evaluation**

Scripts compute:

Accuracy

Precision / Recall / F1

Confusion counts

Model comparisons

**Repository Structure**
- Data/                       # Input & model outputs
- LLM Chats/                  # Generate LLM model predictions
- data_import.py              # Data loaders of the original dataset
- data_import_testdata.py     # Data loaders of the test dataset
- prompt_writing_functions.py # JSON prompt generation
- other_functions.py          # Helper functions
- main.py                     # Zero-shot + FT classification
- prepare_openai_finetune.py  # Create train/val/test datasets
- ft_cod_openai_mini.py       # Fine-tune GPT-4o-mini
- ft_cod_openai.py            # Fine-tune GPT-4o

