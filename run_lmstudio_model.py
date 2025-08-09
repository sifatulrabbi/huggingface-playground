"""
Didn't work, LM Studio specific models requires llama-cpp/llama-cpp-python for inference.
"""

import torch
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.text_generation import TextGenerationPipeline
from langchain_huggingface import HuggingFacePipeline

if torch.mps.is_available():
    print("MPS available")

device = torch.device("mps")
model_path = "/Users/sifatul/.lmstudio/models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-MLX-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)

chain = PromptTemplate.from_template(
    "Write me an algorithm for {topic} in Go."
) | HuggingFacePipeline(pipeline=pipeline)
response = chain.invoke({"topic": "Binary search"})

print(response)
