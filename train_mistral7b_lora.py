import torch
from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding_side="right", use_fast=False
)

if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

data = load_dataset(
    "json", data_files="./datasets/finetune/json_output.json", split="train"
)
