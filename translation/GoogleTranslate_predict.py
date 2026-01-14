import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./SlovenBERTcina/SlovenBERTcina/')

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'sk': '<sk>',
    'cz': '<cz>',
    'ru': '<ru>'
}

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

model = torch.load("./SlovenBERTcina/SlovakToEnglish/skencheckpoint_200000.pth")

input_text = "My name is Chris."


print(f"Raw Input Text: {input_text}")

input_ids = encode_input_str(
    text = input_text,
    target_lang = 'sk',
    tokenizer = tokenizer,
    seq_len = model.config.max_length,
    lang_token_map= LANG_TOKEN_MAPPING
)
input_ids = input_ids.unsqueeze(0).cuda()

output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=3)
for token_set in output_tokens:
    print("Guess: ",tokenizer.decode(token_set, skip_special_tokens=True))
