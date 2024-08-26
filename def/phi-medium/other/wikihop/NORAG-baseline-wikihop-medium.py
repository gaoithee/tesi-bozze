import pandas as pd
import torch
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
import re
import random
from guidance import models, select
from langchain import PromptTemplate
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('wikihop-dataset-1000.csv')

N_examples = len(df)
# select a subset of the queries, just for test:
first_queries = df['cleaned_query']
# same for correct answers and distractors:
correct_answers = df['answer']
possibilities = df['options']
# and for the sources:
sources = df['supports']

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-medium-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct", use_fast=False)
new_model = models.Transformers(model, tokenizer, temperature=0.0)

def clean_text(text):
    return re.sub(r"[^\w\s.,!?\-:;()]+", '', text)

###################################################################

# thesis
def create_message(question, options, context):
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

def thesisGeneration(query, merged, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = create_message(query, merged, sources)
    ans = new_model + str(augmented_prompt) + select(merged)
    return str(ans)

def extract_answer(text):
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

# answers = []
# for i in range(len(df)):
#     answers.append(extract_answer(thesisGeneration(first_queries[i], possibilities[i], sources[i][:1000])))

###################################################################

# antithesis
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

def create_message2(question, candidate, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Context: " + context + "\n Assistant: \n"

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
        Context: The Sun is the star at the center of the Solar System. It is a massive, nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet energies.

        Assistant: The correct answer should be 'a star' due to the fact that the context explicitly say so. On the opposite, the context never mentions the fact that the Sun could be a planet.
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def antithesisGeneration(query, merged, candidate, sources):
    merged = ast.literal_eval(merged)
    prompt = create_message2(query, candidate, merged, sources)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

# ant_answers = []
# for i in range(len(df)):
#     ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], answers[i], sources[i][:1000]))

###################################################################

# synthesis
def create_message3(question, candidate, suggestion, options, context):

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Context: " + context + "\n Assistant: \n"

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
        Context: The Sun is the star at the center of the Solar System. It is a massive, nearly perfect sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet energies.

        Assistant: the correct option is 'a star', since the suggestion is grounded in the context, even if the candidate answer does not agree.
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages


def preSynthGeneration(query, candidate_answer, critique, merged, sources):
    prompt = create_message3(query, merged, candidate_answer, critique, sources)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

# pre_answers = []
# for i in range(len(df)):
#    pre_answers.append(preSynthGeneration(first_queries[i], answers[i], ant_answers[i], possibilities[i], sources[i][:1000]))

###################################################################

# format synthesis
prompt_template = PromptTemplate.from_template(
"""You are a multiple-choice question answering assistant.
Choose the most proper answer between {options} that best matches with the suggestion. 

Question: {question}
Suggestion: {critique}

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
    ans = new_model + normal_string + select(merged)
    return str(ans)

def extract_answer(text):
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

temp = pd.read_csv('def/phi-medium/phi-medium-baseline-wikihop.csv')
pre_answers = temp['pre-synthesis']
answers = temp['thesis']
ant_answers = temp['antithesis']

syn_answers = []
for i in range(len(df)):
    syn_answers.append(extract_answer(
        synthesisGeneration(
            first_queries[i], prompt_template, possibilities[i], 
            pre_answers[i], sources[i][:1000])))

###################################################################


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
df['correct'] = df['correct'].apply(clean_text)
df['thesis'] = df['thesis'].apply(clean_text)
df['synthesis'] = df['synthesis'].apply(clean_text)

df.to_csv('def/phi-medium/phi-medium-baseline-wikihop2.csv')














