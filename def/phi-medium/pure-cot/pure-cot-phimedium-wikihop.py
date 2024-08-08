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

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-medium-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct", use_fast=False)
new_model = models.Transformers(model, tokenizer, temperature=0.0)

#############################################

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

def create_message_antithesis(question, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Context: " + context + "\n Assistant: \n"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer that you are required to check with respect to the question and the relevant context.
        Your goal is to decree which is the most correct answer to the question between the available options.

        Here's an example of how to do it:
        """},
        {"role": "user", "content": """
        Question: Jane's Addiction and Weeping Willows, play which genre of music?
        Options: ['indie', 'rock']
        Candidate answer: rock
        Context: Weeping Willows is a Swedish indie rock group that started in 1995. Jane's Addiction is an American rock band from Los Angeles, formed in 1985. The band consists of Perry Farrell (vocals), Dave Navarro (guitar), Stephen Perkins (drums) and Chris Chaney (bass).

        Assistant: Let's consider the options and check whether or not they are correct. The context mentions that Weeping Willows is a 'Swedish indie rock group' and Jane's Addiction is an 'American rock band'. Both bands are associated with the 'rock' genre, thus the correct answer is 'rock'. ### rock
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def antithesisGeneration(query, merged, sources):
    merged = ast.literal_eval(merged)
    prompt = create_message_antithesis(query, merged, sources)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

# --------------------------------------------------

# prompt augmentation for the (format of the) synthesis:
prompt_template = PromptTemplate.from_template(
"""You are a multiple-choice question answering assistant. 
Extract the answer in the context after the ### symbols and choose the most proper option between {options} that best matches with the suggestion. 

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

# --------------------------------------------------

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

df = pd.read_csv('wikihop_dataset/wikihop-merged-summarized.csv')

# select a subset of the queries, just for test:
first_queries = df['query']

# same for correct answers and distractors:
correct_answers = df['answer']
possibilities = df['options']

# and for the sources:
sources = df['sum_supports']

N_rows = len(df)

#############################################

# ANTITHESIS
ant_answers = []
for i in range(N_rows):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], sources[i]))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'cot': ant_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)

df.to_csv('purecot-phimedium-wikihop.csv')
