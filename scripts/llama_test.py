import os
import torch
from huggingface_hub import login
from transformers import pipeline

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
if hf_token:
    login(token=hf_token)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are an extroverted person."},
    {"role": "user", "content": "Tell me about yourself."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])