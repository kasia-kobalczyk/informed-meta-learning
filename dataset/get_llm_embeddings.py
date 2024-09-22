import pandas as pd
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
import torch


df = pd.read_csv('./data/temperatures/2021-2022_AK_gpt_descriptions.csv')
ds = Dataset.from_pandas(df)

llm_name = "meta-llama/Meta-Llama-3-8B"
device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained(llm_name, add_eos_token=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

model = AutoModel.from_pretrained(
    llm_name,
).to(device)
model.eval()

def get_llm_embedding(examples):
    desc = examples['description']
    inputs = tokenizer(desc, return_tensors='pt')

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        last_hidden_state = model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            output_hidden_states=True
        ).hidden_states[-1]
        embed = last_hidden_state[:, -1, :]
    
    return {
        'LST_DATE' : examples['LST_DATE'],
        'embed' : embed
    }

embeded_ds = ds.map(
    get_llm_embedding,
    batched=False,
)
embeded_ds.save_to_disk(f"./data/temperatures/2021-2022_AK_desc-embeded-llama")