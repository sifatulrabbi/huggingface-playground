import torch
from transformers.pipelines import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from icecream import ic


set_verbosity_error()


device = torch.device("mps")
model = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-0.6B",
    device=device,
    truncation=True,
    max_length=1024 * 10,
)

# response = model(
#     "I am a haiku writer who is expert in writing haikus.\n\nUSER: Write me a haiku about rain.\n\nOkay I now have to write a haiku about rain, here I go.\n"
# )
# ic(response)

chain = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand.\n\n",
) | HuggingFacePipeline(pipeline=model)

topic = input("Topic: ")
age = input("age: ")
stream = chain.stream({"topic": topic, "age": age})
for chunk in stream:
    print(chunk, end="", flush=True)
