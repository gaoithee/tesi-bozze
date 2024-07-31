import pandas as pd
import torch
import ast
import datasets
from datasets import load_dataset
from peft import LoraConfig, PeftModel
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
import re
import random
import warnings
warnings.filterwarnings("ignore")

dataset = load_dataset('saracandu/hotpotQA_nli', split="train", trust_remote_code=True)
N_examples = 100
# select a subset of the queries, just for test:
first_queries = dataset['question'][:N_examples]
# same for correct answers and distractors:
correct_answers = dataset['answer'][:N_examples]
possibilities = dataset['options'][:N_examples]
# and for the sources:
sources = dataset['passages'][:N_examples]

##############################################################################################

model_name="meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    do_sample=False,
    temperature=0.0,
)

new_model = models.Transformers(model, tokenizer, temperature=0.0)

##############################################################################################

def clean_text(text):
    return re.sub(r"[^\w\s.,!?\-:;()]+", '', text)

##############################################################################################

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
    # normal_string = clean_text(augmented_prompt)
    ans = new_model + str(augmented_prompt) + select([clean_text(merged[0]), clean_text(merged[1])])
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

answers = []
for i in range(N_examples):
    answers.append(extract_answer(thesisGeneration(first_queries[i], possibilities[i], sources[i])))

##############################################################################################

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def create_message2(question, candidate, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Context: " + context + "\n Assistant: \n"

    messages = [
        {"role": "user", "content": """
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

        Assistant:
        """
        },
        {"role": "assistant", "content": "The correct answer should be 'a star' due to the fact that the context explicitly say so. On the opposite, the context never mentions the fact that the Sun could be a planet."},
        {"role": "user", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages


def antithesisGeneration(query, merged, candidate, sources):
    merged = ast.literal_eval(merged)
    prompt = create_message2(query, candidate, merged, sources)
    outputs = pipeline(prompt, max_new_tokens=500, eos_token_id=terminators, do_sample=False)
    return outputs[0]["generated_text"][-1]

ant_answers = []
for i in range(N_examples):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], answers[i], sources[i]))

##############################################################################################

def create_message3(question, candidate, suggestion, options, context):

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Context: " + context + "\n Assistant: \n"

    messages = [
        {"role": "user", "content": """
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
        Assistant:
        """
        },
        {"role": "assistant", "content": """
        the correct option is 'a star', since the suggestion is grounded in the context, even if the candidate answer does not agree.
        """
        },
        {"role": "user", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages


def extract_last_assistant_content(pre_answers):
    # Filtra gli elemefrom groq import Groqnti con 'role' uguale a 'assistant'
    assistant_responses = [entry['content'] for entry in pre_answers if entry['role'] == 'assistant']
    # Restituisci il 'content' dell'ultimo elemento filtrato
    return assistant_responses[-1] if assistant_responses else None

def preSynthGeneration(query, candidate_answer, critique, merged, sources):
    prompt = create_message3(query, merged, candidate_answer, critique, sources)
    output = pipeline(prompt, max_new_tokens=500, eos_token_id=terminators, do_sample=False)
    return output[0]['generated_text']

pre_answers = []
pre_answers_def = []

for i in range(N_examples):
    pre_answers.append(preSynthGeneration(first_queries[i], answers[i], ant_answers[i], possibilities[i], sources[i]))
    pre_answers_def.append(extract_last_assistant_content(pre_answers[i]))


##############################################################################################

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
    ans = new_model + normal_string + select([clean_text(merged[0]), clean_text(merged[1])])
    return str(ans)

def extract_answer(text):
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

syn_answers = []
for i in range(N_examples):
    syn_answers.append(extract_answer(
        synthesisGeneration(
            first_queries[i], prompt_template, possibilities[i], 
            pre_answers_def[i], sources[i])))

##############################################################################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': ant_answers,
    'pre-synthesis': pre_answers_def,
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

df.to_csv('llama3-instruct-baseline.csv')
