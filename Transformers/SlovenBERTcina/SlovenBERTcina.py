#from datasets import load_dataset

#from pathlib import Path
import os

import torch
print(f"We have cuda: {torch.cuda.is_available()}")

from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# dataset = load_dataset("oscar", "unshuffled_deduplicated_sk")
# #paths = [str(x) for x in Path(".").glob("**/*.txt")]
# with open("SKDataset_part_1.txt", "w", encoding="utf-8") as f:
#     i = 0
#     for line in dataset["train"][:]["text"]:
#         if i < 100000:
#             f.write(line)
#         i += 1

# Customize training
tokenizer.train(files="SKDataset.txt", vocab_size=500000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
token_dir = '/SlovenBERTcina'
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model('./SlovenBERTcina')
torch.cuda.empty_cache()

tokenizer = ByteLevelBPETokenizer(
    "./SlovenBERTcina/vocab.json",
    "./SlovenBERTcina/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(
    tokenizer.encode("Volám sa Chris, tesí ma.").tokens
)
torch.cuda.empty_cache()

# Config file for RoBERTA transformers
config = RobertaConfig(
    vocab_size=500000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Initialize Our tokenizer as the pretrained one
tokenizer = RobertaTokenizer.from_pretrained("./SlovenBERTcina", max_length=512)

# Initialize our Model
model = RobertaForMaskedLM(config=config)

print(f" Number of trained model parameters: {model.num_parameters()}")

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./SKDataset.txt",
    block_size=128,
)

# Define a Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Initialize a Trainer
training_args = TrainingArguments(
    output_dir="./SlovenBERTcina",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

trainer.save_model("./SlovenBERTcina")

fill_mask = pipeline(
    "fill-mask",
    model="./SlovenBERTcina",
    tokenizer="./SlovenBERTcina"
)

print(fill_mask("Mnoho ľudí tu<mask>."))
print(fill_mask("Ako sa<mask>"))
print(fill_mask("Plážová sezóna pod Zoborom patrí medzi<mask> obdobia."))