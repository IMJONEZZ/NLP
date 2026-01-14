#from transformers import RobertaModel, RobertaTokenizer
from transformers import BertModel, BertTokenizer

#model_version = './SlovenBERTcina'
model_version = 'bert-base-uncased'
do_lower_case = True

# model = RobertaModel.from_pretrained(model_version, output_attentions=True)
# tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

sen_a = "The cat sleeps on the ground"
sen_b = "Mačka spí na zemi"
inputs = tokenizer.encode_plus(sen_a, sen_b, return_tensors='pt', add_special_tokens=True, return_token_type_ids=True)

#print(inputs)

token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']

attention = model(input_ids, token_type_ids=token_type_ids)[-1]
input_id_list = input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

print(tokens)