import pandas as pd
import torch
import ast
import re
import random
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from guidance import models, select
from langchain import PromptTemplate
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

#############################################

# Definisci una funzione di pulizia per rimuovere caratteri non validi
def clean_text(text):
    return re.sub(r"[^\w\s.,!?\-:;()]+", '', text)


def create_message_thesis(question, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Answer to the following question: " + question + " providing one of these options as answer: " + str(options) + "Assistant:"

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You have to provide helpful answers to the user’s questions based on the context: 
        """ + context},
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


def thesisGeneration(query, merged, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = create_message_thesis(query, merged, sources)
    ans = new_model + str(augmented_prompt) + select([clean_text(merged[0]), clean_text(merged[1])])
    return str(ans)


def preSynthGeneration(query, merged, candidate_answer, critique, sources):
    merged = ast.literal_eval(merged)
    second_answer = query_model(system_message.format(context = sources),
    user_message.format(question=query, options = merged, candidate_answer = candidate_answer, critique=critique, context = sources,), max_length=1024)
    return second_answer


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

def synthesisGeneration(query, prompt_template, merged, pre_answer, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'question': query, 
                                            'options': merged,
                                            'critique': pre_answer,
                                            'context': sources})

    normal_string = clean_text(augmented_prompt.text)
    ans = new_model + normal_string + select([clean_text(merged[0]), clean_text(merged[1])])
    return str(ans)

def extract_answer_synthesis(text):
    # Trova l'indice in cui inizia il testo "Why or why not the answer is correct:"
    start_index = text.find("assistant:\n")

    
    # Se l'indice è stato trovato, estrai la risposta corretta
    if start_index != -1:
        start_index += len("assistant:\n")
        # Estrai il testo dopo "Why or why not the answer is correct:"
        correct_answer_text = text[start_index:].strip()
        return correct_answer_text
    else:
        return "The correct answer could not be found."

def remove_brackets(s):
    return s.strip("[] ")

# Definisci una funzione di pulizia per rimuovere caratteri non validi
def clean_text_final(text):
    text = re.sub(r'[^\w\s.,!?\'"\-:;()]+', '', text)  # Rimuove i caratteri speciali
    text = re.sub(r"['\"-]", '', text)  # Rimuove apostrofi, virgolette e trattini
    text = text.lower()  # Converte in minuscolo
    return text

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

#############################################

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", use_fast=False)
new_model = models.Transformers(model, tokenizer, temperature=0.0)

pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.5,
    return_full_text=True,
    max_new_tokens=400,
    top_p=0.0
)

# prompt per baseline-2
system_message = """
    You are an helpful AI assistant.
    You are asked to determine the most correct answer for a given question.
    You have at disposal a first tentative answer (a candidate answer) and another opinion on which should be the correct option according to context (a suggestion).
    
    They could agree on the correct option; in this case, directly output the option on which they agree.
    If instead they disagree, use the context to determine the correct answer for the question, given the set of possible options.
    
    The goal of the assistant is to decree which is the most correct answer to the question between the available options. 
    Answer by explicitly reporting the correct answer to you.
"""


user_message = """
    Question: {question}
    Options: {options}
    Candidate answer: {candidate_answer}
    Suggestion: {critique}
    Which of the candidate answers {options} is the most proper answer for the question?

"""

def query_model(
        system_message,
        user_message,
        temperature = 0.0,
        max_length=1024
        ):

    user_message = "Question: " + user_message + " Correct answer:"
    messages = [
        {"role": "System", "content": system_message},
        {"role": "User", "content": user_message},
        ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    sequences = pipeline(
        prompt,
        top_p=0.0,
        temperature=temperature,
        #num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.model.config.eos_token_id
    )

    answer = sequences[0]['generated_text']
    return answer 
    
#############################################

# THESIS
answers = []
for i in range(len(first_queries)):
    answers.append(extract_answer_thesis(thesisGeneration(first_queries[i], possibilities[i], sources[i])))


# ANTITHESIS
bart_answers = []
for i in range(len(first_queries)):
    if bart1[i] > bart2[i]:
        bart_answers.append(first_nli[i])
    else:
        bart_answers.append(second_nli[i])


# SYNTHESIS
pre_answers = []
for i in range(len(first_queries)):
    pre_answers.append(preSynthGeneration(first_queries[i], possibilities[i], answers[i], bart_answers[i], sources[i]))


# format synthesis
syn_answers = []
for i in range(len(first_queries)):
    syn_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], prompt_template, possibilities[i], 
            bart_answers[i], sources[i])))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': bart_answers,
    'pre-synthesis': pre_answers,
    'synthesis': syn_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)
df['synthesis'] = df['synthesis'].apply(clean_text_final)

df.to_csv('def/phi-mini/bart-naive.csv')