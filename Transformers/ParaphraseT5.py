from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = model.to(device)

sentence = "Do you know of anyone who has experienced COVID19 symptoms in the past two weeks"

text = "paraphrase: " + sentence.lower() + " </s>"

encoding = tokenizer.encode_plus(text,pad_to_max_length=True,return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=120,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)
