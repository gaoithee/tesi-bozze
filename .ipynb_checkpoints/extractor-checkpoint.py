import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from guidance import models, select
import pandas as pd
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
import ast
import re

mini = pd.read_csv('PEC-phimedium-ctx-bridge.csv')

# prompt augmentation for the (format of the) synthesis:
prompt_template = PromptTemplate.from_template(
"""You are a multiple-choice question answering assistant.
Choose the most proper option between {options} that best matches with the suggestion. 

Question: {question}
Context: {critique}
Sources: {context}

Assistant:
"""
)
augmentation = {"question": itemgetter("question"),
                "options": itemgetter("options"), 
                "critique": itemgetter("critique"),
                "context": itemgetter("context"), }
synthesis_chain = augmentation | prompt_template 

def clean_text(text):
    return re.sub(r"[^\w\s.,!?\-:;()]+", '', text)

def synthesisGeneration(query, merged, pre_answer, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'question': query, 
                                            'options': merged,
                                            'critique': pre_answer,
                                            'context': sources})

    normal_string = clean_text(augmented_prompt.text)
    ans = new_model + normal_string + select(merged)
    return str(ans)

def extract_answer_synthesis(text):
    # Trova l'indice in cui inizia il testo "Why or why not the answer is correct:"
    start_index = text.find("\n\nAssistant:\n")

    
    # Se l'indice è stato trovato, estrai la risposta corretta
    if start_index != -1:
        start_index += len("\n\nAssistant:\n")
        # Estrai il testo dopo "Why or why not the answer is correct:"
        correct_answer_text = text[start_index:].strip()
        return correct_answer_text
    else:
        return "The correct answer could not be found."

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=False)
new_model = models.Transformers(model, tokenizer, temperature=0.0)

df = pd.read_csv('ultramega-test.csv')

# select a subset of the queries, just for test:
first_queries = df['question']

# same for correct answers and distractors:
correct_answers = df['answer']
possibilities = df['options']

# and for the sources:
sources = df['new']

N_rows = len(df)

syn_answers = []
for i in range(len(df)):
    syn_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            mini['pre-synthesis'][i], sources[i])))

ant_answers = []
for i in range(len(df)):
    ant_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            mini['antithesis'][i], sources[i])))

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': mini['thesis'],
    'pre-antithesis': mini['antithesis'],
    'antithesis': ant_answers,
    'pre-synthesis': mini['pre-synthesis'],
#    'cot': mini['cot'],
    'synthesis': syn_answers,
    'context': sources
} 

df = pd.DataFrame(df)

def clean_text_final(text):
    text = re.sub(r'[^\w\s.,!?\'"\-:;()]+', '', text)  # Rimuove i caratteri speciali
    text = re.sub(r"['\"-]", '', text)  # Rimuove apostrofi, virgolette e trattini
    text = text.lower()  # Converte in minuscolo
    return text

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)
df['antithesis'] = df['antithesis'].apply(clean_text_final)
df['synthesis'] = df['synthesis'].apply(clean_text_final)

df.to_csv('PEC-phimedium-ctx-bridge.csv')