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

fullwiki = load_dataset('saracandu/my_wikihop', split='train')

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
}

def create_message(question, options):
    options_str = '", "'.join(options)
    content = f"""
    You are asked to create a question out of a sketched question.

    Here's an example:
    Question: "occupation cao chong"
    Options: ['academic', 'builder', 'chancellor', 'classics', 'confucian scholar', 'designer', 'duke', 'emperor', 'engineer', 'engineering', 'father', 'founder', 'general', 'king', 'leader', 'major', 'mathematician', 'military', 'official', 'peasant', 'physicist', 'physics', 'politician', 'prior', 'rebel', 'research', 'ruler', 'science', 'script', 'social reformer', 'socialist', 'sovereign', 'taiwan']
    Assistant: "Which was the occupation of Cao Chong?"

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": content}
    ]
    
    return messages


dataset = load_dataset('saracandu/my_wikihop', split='train')

questions = dataset['query']
options = dataset['candidates']

prompts = []
outs = []
for i in range(4000):
  options[i] = ast.literal_eval(options[i])
  prompts.append(create_message(questions[i], options[i]))
  output = pipe(prompts[i], **generation_args)
  outs.append(output[0]['generated_text'])

def extract_questions(statements):
    questions = []
    for statement in statements:
        # Usa una regex per trovare le domande
        match = re.search(r'\"(.*?)\"', statement)
        if match:
            questions.append(match.group(1))
        else:
            # Se non ci sono virgolette, prendi tutta la frase
            questions.append(statement.strip(' "'))
    return questions

dati = {
    'query': outs[:4000],
    'answer': dataset['answer'][:4000],
    'options': dataset['candidates'][:4000],
    'supports': dataset['supports'][:4000],
    
}

df = pd.DataFrame(dati)
df.to_csv('wikihop-4000.csv')
