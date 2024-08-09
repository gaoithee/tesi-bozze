import pandas as pd
import torch
import re
import ast
import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import warnings
warnings.filterwarnings("ignore")

dataset = load_dataset('saracandu/my_wikihop', split='train')
sources = dataset['supports']

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", use_fast = False)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def produce_prompt(source):
    messages = [ 
        {"role": "system", "content": """
        You are a helpful AI assistant. You are given a long document as context and you are asked to produce a meaningful summary of it.
        Please summarize in less than 500 words.
        """}, 
        {"role": "user", "content": source},
    ] 
    return messages

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 1000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

summ = []
for i in range(300):
    output = pipe(produce_prompt(sources[i]), **generation_args) 
    summ.append(output[0]['generated_text']) 
    torch.cuda.empty_cache()
    
queries = dataset['cleaned_query'][600:900]
answers = dataset['answer'][600:900]
options = dataset['options'][600:900]

dati = {'query': queries, 'answer': answers, 'options': options, 'sum_supports': summ}
df = pd.DataFrame(dati)
df.to_csv('save-first601-900.csv')




