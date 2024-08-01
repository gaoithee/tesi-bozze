import pandas as pd
import ast
import re
import torch
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

# for generating the 'thought' of the synthesis
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

# --------------------------------------------------

#############################################

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

def extract_thesis(text):
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
    ans = new_model + str(augmented_prompt) + select(merged)
    return extract_thesis(str(ans))

#############################################

def create_message_antithesis(question, candidate, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Context: " + context + "\n\n Assistant:"

    messages = [
        {"role": "user", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer that you are required to check with respect to the question and the relevant context.
        Your goal is to decree which is the most correct answer to the question between the available options.

        Here's an example of how to do it:
        Question: What is the sun, a star or a planet?
        Options: ['a star', 'a planet']
        Candidate answer: a planet
        Context: The Sun is the star at the center of the Solar System. It is a massive, nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet energies.
        """},
        {"role": "assistant", "content": """
        The correct answer should be 'a star' due to the fact that the context explicitly say so. On the opposite, the context never mentions the fact that the Sun could be a planet.
        """
        },
        {"role": "user", "content": "Now do the same for the following question: \n" + user_content}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_antithesis(text):
    pattern = re.compile(r'<start_of_turn>model(.*?)<end_of_turn>', re.DOTALL)
    matches = pattern.findall(text)
    
    if matches:
        # Prendi l'ultimo match
        extracted_text = matches[-1]
        # Rimuovi i simboli "_"
        cleaned_text = extracted_text.replace('▁', '').strip()
        return cleaned_text
    else:
        return None

def antithesisGeneration(query, merged, candidate, sources):
    merged = ast.literal_eval(merged)
    prompt = create_message_antithesis(query, candidate, merged, sources)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=500)
    return extract_antithesis(tokenizer.decode(outputs[0]))

#############################################

def create_message_presynthesis(question, candidate, suggestion, options, context):
    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Context: " + context 
    chat = [
            {"role": "user", "content": """
            You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
            You also have at disposal a first tentative answer that you are required to check with respect to the question and the relevant context.
            Your goal is to decree which is the most correct answer to the question between the available options.
    
            Here's an example of how to do it:
            Question: What is the sun, a star or a planet?
            Options: ['a star', 'a planet']
            Candidate answer: a planet
            Context: The Sun is the star at the center of the Solar System. It is a massive, nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet energies.
            """},
            {"role": "assistant", "content": """
            The correct answer should be 'a star' due to the fact that the context explicitly say so. On the opposite, the context never mentions the fact that the Sun could be a planet.
            """
            },
            {"role": "user", "content": "Now do the same for the following question: "+ user_content}
        ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def preSynthGeneration(query, candidate_answer, critique, merged, sources):
    prompt = create_message_presynthesis(query, merged, candidate_answer, critique, sources)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=500)
    return extract_antithesis(tokenizer.decode(outputs[0]))

#############################################

def synthesisGeneration(query, merged, pre_answer, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'question': query, 
                                            'options': merged,
                                            'critique': pre_answer,
                                            'context': sources})

    normal_string = clean_text(augmented_prompt.text)
    ans = new_model + normal_string + select(merged)
    return extract_synthesis(str(ans))

def extract_synthesis(text):
    # Trova l'indice in cui inizia il testo "Why or why not the answer is correct:"
    start_index = text.find("\nAssistant:\n")

    
    # Se l'indice è stato trovato, estrai la risposta corretta
    if start_index != -1:
        start_index += len("\nAssistant:\n")
        # Estrai il testo dopo "Why or why not the answer is correct:"
        correct_answer_text = text[start_index:].strip()
        return correct_answer_text
    else:
        return "The correct answer could not be found."

#############################################

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", token = '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', use_fast = False)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    token = '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',
    torch_dtype=torch.bfloat16
)

new_model = models.Transformers(model, tokenizer, temperature=0.0)

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
    answers.append(thesisGeneration(first_queries[i], possibilities[i], sources[i]))


# ANTITHESIS
ant_answers = []
for i in range(N_rows):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], answers[i], sources[i]))

# SYNTHESIS
pre_answers = []
for i in range(N_rows):
    pre_answers.append(preSynthGeneration(first_queries[i], possibilities[i], answers[i], ant_answers[i], sources[i]))


# format synthesis
syn_answers = []
for i in range(N_rows):
    syn_answers.append(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            pre_answers[i], sources[i]))


def_answers = ["The correct option is " + clean_text(correct_answer) + " due to what is said in the context." for correct_answer in correct_answers]

# format synthesis
oracle_answers = []
for i in range(N_rows):
    oracle_answers.append(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            def_answers[i], sources[i]))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': ant_answers,
    'pre-synthesis': pre_answers,
    'synthesis': syn_answers,
    'oracle': oracle_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Funzione per rimuovere le quadre e ottenere solo il contenuto
def remove_brackets(s):
    return s.strip("[] ")

# Definisci una funzione di pulizia per rimuovere caratteri non validi
def clean_text(text):
    text = re.sub(r'[^\w\s.,!?\'"\-:;()]+', '', text)  # Rimuove i caratteri speciali
    text = re.sub(r"['\"-]", '', text)  # Rimuove apostrofi, virgolette e trattini
    text = text.lower()  # Converte in minuscolo
    return text

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)
df['synthesis'] = df['synthesis'].apply(clean_text_final)
df['oracle'] = df['oracle'].apply(clean_text_final)


df.to_csv('baseline-gemma-2-9b-it.csv')


