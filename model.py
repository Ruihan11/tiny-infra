import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Start loading model ...")

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto")

print("Finished loading have fun ヽ( ຶ▯ ຶ)ﾉ!!!")
print(f"devices : {model.device}")
print(f"model para size : {model.num_parameters()/1e9:.2f}B")










