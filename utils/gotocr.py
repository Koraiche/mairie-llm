# GOT OCR
#!pip install tiktoken --quiet
#!pip install verovio --quiet
#pip install accelerate
from transformers import AutoModel, AutoTokenizer
import torch

device = torch.device("cpu")

# Initialiser GOT OCR
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id).to(device)

# plain texts OCR
res = model.chat(tokenizer, "./image.png", ocr_type='ocr')
print(res)