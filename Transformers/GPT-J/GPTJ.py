from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
CONTEXT = str(input("Enter the context: "))

input_ids = tokenizer(CONTEXT, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length = 256)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(CONTEXT, gen_text)