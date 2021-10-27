from datasets import load_dataset
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup

model_repo = 'google/mt5-base'

tokenizer = AutoTokenizer.from_pretrained(model_repo)

model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
model.config.max_length = 40
model = model.cuda()

input_sent = 'Here is a test sentence!'
token_ids = tokenizer.encode(input_sent, return_tensors='pt').cuda()

model_out = model.generate(token_ids)

output_text = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(model_out[0])
)

#print(output_text)

example_input_str = '<sk>This is a test znsilof.'
input_ids = tokenizer.encode(example_input_str, return_tensors='pt')
#print('Input ids: ',input_ids)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#print('Tokens: ', tokens)

dataset = load_dataset('text', data_files='./SlovenBERTcina/en-sk.txt')

# train_dataset, test_dataset = train_test_split(dataset, shuffle=True, test_size = 0.2)
train_dataset = dataset['train'].shuffle()

#print(train_dataset[0]['text'])

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'sk': '<sk>',
    'cz': '<cz>',
    'ru': '<ru>'
}
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

token_ids = tokenizer.encode(
    example_input_str, 
    return_tensors='pt', 
    padding='max_length',
    truncation=True,
    max_length=model.config.max_length
)

def encode_input_str(text, target_lang, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING):
    target_lang_token = lang_token_map[target_lang]

    input_ids = tokenizer.encode(
        text = target_lang_token + text,
        return_tensors = 'pt',
        padding = 'max_length',
        truncation = True,
        max_length = seq_len
    )
    return input_ids[0]

def encode_target_str(text, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING):
    token_ids = tokenizer.encode(
        text = text,
        return_tensors = 'pt',
        padding = 'max_length',
        truncation = True,
        max_length = seq_len
    )
    return token_ids[0]

def format_translation_data(translations, lang_token_map, tokenizer, seq_len=128):
    input_lang, target_lang = 'en', 'sk'
    
    x = re.search("[\t]", translations)
    start = x.start()
    end = x.end()

    input_text = translations[:start]
    #print(input_text)
    target_text = translations[end:]
    #print(target_text)

    if input_text is None or target_text is None:
        return None

    input_token_ids = encode_input_str(
        input_text, target_lang, tokenizer, seq_len, lang_token_map
    )
    target_token_ids = encode_target_str(
        target_text, tokenizer, seq_len, lang_token_map
    )

    return input_token_ids, target_token_ids

def transform_batch(batch, lang_token_map, tokenizer):
    inputs = []
    targets = []
    for translation_set in batch:
        formatted_data = format_translation_data(
            translation_set, lang_token_map, tokenizer
        )
        if formatted_data is None:
            continue
        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))
    batch_input_ids = torch.cat(inputs).cuda()
    batch_target_ids = torch.cat(targets).cuda()

    return batch_input_ids, batch_target_ids

def get_data_generator(dataset, lang_token_map, tokenizer, batch_size=32):
    dataset = dataset.shuffle()
    for i in range(0, len(dataset), batch_size):
        raw_batch = dataset[i:i+batch_size]['text']
        yield transform_batch(raw_batch, lang_token_map, tokenizer)

in_ids, out_ids = format_translation_data(
    train_dataset[0]['text'], LANG_TOKEN_MAPPING, tokenizer
)

#print(" ".join(tokenizer.convert_ids_to_tokens(in_ids)))
#print(" ".join(tokenizer.convert_ids_to_tokens(out_ids)))


n_epochs = 1
batch_size = 4
print_freq = 50
check_freq = 500
checkpoint_freq = 5000
lr = 5e-4
n_batches = int(np.ceil(len(train_dataset) / batch_size))
total_steps = n_epochs * n_batches
n_warmup_steps = int(total_steps * 0.01)
print(f"TOTAL NUMBER OF STEPS: {total_steps}")

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, n_warmup_steps, total_steps
)

losses = []

for epoch_idx in range(n_epochs):
    data_generator = get_data_generator(train_dataset, LANG_TOKEN_MAPPING, tokenizer, batch_size)

    for batch_idx, (input_batch, label_batch) in enumerate(data_generator):
        optimizer.zero_grad()

        if (batch_idx + 1) % check_freq == 0:
            print(f"Input: {input_batch[0]}")
            print(f"Label: {label_batch[0]}")

        model_out = model.forward(
            input_ids = input_batch,
            labels = label_batch
        )
        if (batch_idx + 1) % check_freq == 0:
            print(f"Model Guess: {model_out[0]}")
            

        loss = model_out.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % print_freq == 0:
            avg_loss = np.mean(losses[-print_freq:])
            print(f"Epoch: {epoch_idx+1} | Step: {batch_idx+1} | Avg. Loss: {avg_loss} | lr: {scheduler.get_last_lr()[0]}")

        if (batch_idx + 1) % checkpoint_freq == 0:
            torch.save(model, f"./SlovenBERTcina/SlovakToEnglish/skencheckpoint_{batch_idx + 1}.pth")

    
torch.save(model, f"./SlovenBERTcina/SlovakToEnglish/skencheckpoint_{batch_idx + 1}.pth")