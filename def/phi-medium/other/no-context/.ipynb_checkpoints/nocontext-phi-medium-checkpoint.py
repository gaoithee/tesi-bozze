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
Choose the most proper option between {options} that best matches with the suggestion. 

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

def create_message_thesis(question, options):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Answer to the following question: " + question + " providing one of these options as answer: " + str(options) + "Assistant:"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You have to provide helpful answers to the user’s questions. 
        """},
        {"role": "user", "content": user_content}
    ]

    return messages

def extract_answer_thesis(text):
    # Trova l'indice in cui inizia il testo "Why or why not the answer is correct:"
    start_index = text.find("}]")

    
    # Se l'indice è stato trovato, estrai la risposta corretta
    if start_index != -1:
        start_index += len("}]")
        # Estrai il testo dopo "Why or why not the answer is correct:"
        correct_answer_text = text[start_index:].strip()
        return correct_answer_text
    else:
        return "The correct answer could not be found."

def thesisGeneration(query, merged):
    merged = ast.literal_eval(merged)
    augmented_prompt = create_message_thesis(query, merged)
    ans = new_model + str(augmented_prompt) + select(merged)
    return str(ans)

#############################################

def create_message_antithesis(question, candidate, options):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Assistant: \n"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer that you are required to check with respect to the question and the relevant context.
        Your goal is to decree which is the most correct answer to the question between the available options.

        Here's an example of how to do it:
        """},
        {"role": "user", "content": """
        Question: What is the sun, a star or a planet?
        Options: ['a star', 'a planet']
        Candidate answer: a planet

        Assistant: The correct answer should be 'a star' since it is surely not a planet.
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def antithesisGeneration(query, merged, candidate):
    merged = ast.literal_eval(merged)
    prompt = create_message_antithesis(query, candidate, merged)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

def create_message_presynthesis(question, candidate, suggestion, options):

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Assistant: \n"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer and a suggestion on which is the correct answer.
        Your goal is to decree which is the most correct answer to the question between the available options according to the context.

        Here's an example of how to do it:
        """},
        {"role": "user", "content": """
        Question: What is the sun, a star or a planet?
        Options: ['a star', 'a planet']
        Candidate answer: a planet
        Suggestion: a star is the correct option since the context clearly specifies that the Sun is the star at the center of the Solar System

        Assistant: the correct option is 'a star', since the suggestion is grounded in the context, even if the candidate answer does not agree.
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def preSynthGeneration(query, candidate_answer, critique, merged):
    prompt = create_message_presynthesis(query, merged, candidate_answer, critique)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

def synthesisGeneration(query, merged, pre_answer):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'question': query, 
                                            'options': merged,
                                            'critique': pre_answer})

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

#nli
first_nli = dataset['first nli']
second_nli = dataset['second nli']

bart1 = dataset['BART1']
bart2 = dataset['BART2']

rob1 = dataset['ROBERTA1']
rob2 = dataset['ROBERTA2']

N_rows = len(dataset)

#############################################

# THESIS
answers = []
for i in range(N_rows):
    answers.append(extract_answer_thesis(thesisGeneration(first_queries[i], possibilities[i])))


# ANTITHESIS
ant_answers = []
for i in range(N_rows):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], answers[i]))

# SYNTHESIS
pre_answers = []
for i in range(N_rows):
    pre_answers.append(preSynthGeneration(first_queries[i], possibilities[i], answers[i], ant_answers[i]))


# format synthesis
syn_answers = []
for i in range(N_rows):
    syn_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], pre_answers[i])))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': ant_answers,
    'pre-synthesis': pre_answers,
    'synthesis': syn_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)
df['synthesis'] = df['synthesis'].apply(clean_text_final)

df.to_csv('phi-medium-nocontext.csv')
