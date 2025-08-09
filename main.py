import torch
from transformers.pipelines import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

set_verbosity_error()

modelpath = "/Users/sifatul/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"

device = torch.device("mps")
model = pipeline(
    task="text-generation",
    model=modelpath,
    device=device,
    truncation=True,
    max_length=1024 * 10,
)

chain = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand.\n\n",
) | HuggingFacePipeline(pipeline=model)

topic = input("Topic: ")
age = input("age: ")
stream = chain.stream({"topic": topic, "age": age})
for chunk in stream:
    print(chunk, end="", flush=True)
