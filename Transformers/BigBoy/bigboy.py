import torch
import torch.nn.functional as F
from torch import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import DiffusionPipeline

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

prompt = "If Elon Musk were smart"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
# Beams: take out if you want text more repetetive and to make less sense num_beams=int
# Sampling: Improves diversity of text, set to false if less diversity is wanted do_sample=bool
# Temperature: scales probability of words being generated, T<1 means it will do more sane words, T>1 will do more insane words temperature=float
# Top K and P: restricts the number of possible words to sample from at each step, higher=more unlikely words top_k=int, top_p=float
# No Repeat n-gram size: Punishes the model for picking the same words that have already been picked and sets a cap for those words
output = model.generate(input_ids, max_length=128, num_beams=5, do_sample=True, temperature=0.5, top_k=50, top_p=0.90, no_repeat_ngram_size=2)

#Better about both repetition and stability
def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label

def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:]
        )
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.to("cpu").numpy()

#logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
out = tokenizer.decode(output[0])
#print(out)
texts = out.split(".")
#print(f"\nlog-prob: {logp:.2f}")

for text in texts:
    prompt = text
    images = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]

    for idx, image in enumerate(images):
        image.save(f"{prompt}-{idx}.png")