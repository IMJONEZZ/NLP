from pathlib import Path
import os

import torch
print(f"We have cuda: {torch.cuda.is_available()}")

from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# First we have to load a batch of txts
# paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Next we have to train a tokenizer
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files="./poe.txt", vocab_size=55000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

token_dir = '/PoeBert'
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model('./PoeBert')

tokenizer = ByteLevelBPETokenizer(
    "./PoeBert/vocab.json",
    "./PoeBert/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Config file for RoBERTA transformers
config = RobertaConfig(
    vocab_size=55000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Initialize Our tokenizer as the pretrained one
tokenizer = RobertaTokenizer.from_pretrained("./PoeBert", max_length=512)

# Initialize our Model
model = RobertaForMaskedLM(config=config)

print(f" Number of trained model parameters: {model.num_parameters()}")

# Build a dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./poe.txt",
    block_size=128,
)

# Define a Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Initialize a Trainer
training_args = TrainingArguments(
    output_dir="./PoeBert",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./PoeBert")

fill_mask = pipeline(
    "fill-mask",
    model="./PoeBert",
    tokenizer="./PoeBert"
)

sentence = "Said the raven<mask>."

#NUM_GENERATE = 50

# for i in range(NUM_GENERATE):
#     res = fill_mask(sentence+"<mask>.")
#     #print(res[0]['sequence'])
#     new = res[0]['sequence'][:-1]
#     sentence = new

print(fill_mask(sentence))