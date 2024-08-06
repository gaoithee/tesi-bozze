import pandas as pd
import ast
import re
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from guidance import models, select
from langchain_core.prompts import PromptTemplate
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

#############################################

# Definisci una funzione di pulizia per rimuovere caratteri non validi
def clean_text(text):
    return re.sub(r"[^\w\s.,!?\-:;()]+", '', text)

# Definisci una funzione di pulizia per rimuovere caratteri non validi
def clean_text_final(text):
    text = re.sub(r'[^\w\s.,!?\'"\-:;()]+', '', text)  # Rimuove i caratteri speciali
    text = re.sub(r"['\"-]", '', text)  # Rimuove apostrofi, virgolette e trattini
    text = text.lower()  # Converte in minuscolo
    return text

#############################################

# prompts and similar things:

# --------------------------------------------------

# prompt augmentation for the (format of the) synthesis:
prompt_template = PromptTemplate.from_template(
"""You are a multiple-choice question answering assistant. 
Extract the answer in the context after the ### symbols and choose the most proper option between {options} that best matches with the suggestion. 

Question: {question}
Context: {critique}

Assistant:
"""
)
augmentation = {"question": itemgetter("question"),
                "options": itemgetter("options"), 
                "critique": itemgetter("critique"),}
synthesis_chain = augmentation | prompt_template 

#############################################

def create_message_antithesis(question, options):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Assistant: \n"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        Your goal is to decree which is the most correct answer to the question between the available options.

        Here's an example of how to do it:
        """},
        {"role": "user", "content": """
        Question: What is the sun, a star or a planet?
        Options: ['a star', 'a planet']
        """
        },
        {"role": "assistant", "content": """
        Assistant: Let's consider the options and check whether or not they are correct. The Sun is the star at the center of the Solar System, thus 'a star' is probably the correct option. On the opposite, 'a planet' is unlikely to be the correct option. Therefore, the correct option is 'a star'. ### a star
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def antithesisGeneration(query, merged):
    merged = ast.literal_eval(merged)
    prompt = create_message_antithesis(query, merged)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

def synthesisGeneration(query, merged, pre_answer):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'question': query, 
                                            'options': merged,
                                            'critique': pre_answer,})

    normal_string = clean_text(augmented_prompt.text)
    ans = new_model + normal_string + select(merged)
    return str(ans)

def extract_answer_synthesis(text):
    # Trova l'indice in cui inizia il testo "Why or why not the answer is correct:"
    start_index = text.find("Assistant:\n")

    
    # Se l'indice è stato trovato, estrai la risposta corretta
    if start_index != -1:
        start_index += len("Assistant:\n")
        # Estrai il testo dopo "Why or why not the answer is correct:"
        correct_answer_text = text[start_index:].strip()
        return correct_answer_text
    else:
        return "The correct answer could not be found."

#############################################

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-medium-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct", use_fast=False)
new_model = models.Transformers(model, tokenizer, temperature=0.0)

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

#############################################

dataset = load_dataset('saracandu/hotpotQA_nli', split="train", trust_remote_code=True)

# select a subset of the queries, just for test:
first_queries = dataset['question']

# same for correct answers and distractors:
correct_answers = dataset['answer']
possibilities = dataset['options']

# and for the sources:
sources = dataset['passages']

N_rows = len(dataset)

#############################################

# ANTITHESIS
ant_answers = []
for i in range(N_rows):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i],))

# format synthesis
syn_answers = []
for i in range(N_rows):
    syn_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            ant_answers[i])))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'cot': ant_answers,
    'answer': syn_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['answer'] = df['answer'].apply(clean_text_final)

df.to_csv('nocontext-phimedium-purecot.csv')
