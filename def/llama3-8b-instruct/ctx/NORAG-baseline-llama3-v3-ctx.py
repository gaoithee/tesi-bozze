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

# prompt augmentation for the (format of the) synthesis:
prompt_template = PromptTemplate.from_template(
"""You are an information extractor assistant.
Choose the most proper option between {options} that best matches with the last statement inside the suggestion.

For example: 
Suggestion: The candidate answer is correct. The context states that Kings of Leon is an American rock band, while The New Pornographers is a Canadian indie rock band. Therefore, the correct answer is no.
Assistant: 'no'

Suggestion: {suggestion}

Assistant:
"""
)
augmentation = {"options": itemgetter("options"), 
                "suggestion": itemgetter("suggestion")
               }
synthesis_chain = augmentation | prompt_template 

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
    ans = new_model + str(augmented_prompt) + select(merged)
    return str(ans)

#############################################

def create_message_antithesis(question, candidate, options, context):
    options_str = '", "'.join(options)
    content = f"""

    Now do the same for this question: "{question}", where options: ["{options_str}"]. Assistant:
    """

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Context: " + context + "\n Assistant: "

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
        Assistant: 
        """
        },
        {"role": "assistant", "content": """
        The context mentions that Weeping Willows is a 'Swedish indie rock group' and Jane's Addiction is an 'American rock band'. Both bands are associated with the 'rock' genre, thus the correct answer is 'rock'.
        """
        },
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def antithesisGeneration(query, merged, candidate, sources):
    merged = ast.literal_eval(merged)
    prompt = create_message_antithesis(query, candidate, merged, sources)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

def create_message_presynthesis(question, candidate, suggestion, options, context):

    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Context: " + context + "\n Assistant: "

    messages = [
        {"role": "system", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer and a suggestion on which is the correct answer.
        Your goal is to decree which is the most correct answer to the question between the available options according to the context.

        Here's a few examples on how to do it:
        """},
        {"role": "user", "content": """
        Question: Jane's Addiction and Weeping Willows, play which genre of music?
        Options: ['indie', 'rock']
        Candidate answer: rock
        Suggestion: The context mentions that Weeping Willows is a 'Swedish indie rock group' and Jane's Addiction is an 'American rock band'. Both bands are associated with the 'rock' genre, thus the correct answer is 'rock'.
        Context: Weeping Willows is a Swedish indie rock group that started in 1995. Jane's Addiction is an American rock band from Los Angeles, formed in 1985. The band consists of Perry Farrell (vocals), Dave Navarro (guitar), Stephen Perkins (drums) and Chris Chaney (bass).
        """
        },
        {"role": "assistant", "content": """
        Assistant: Both the candidate answer and the suggestion agree on the fact that the correct option is 'rock'. Let's check on the context whether or not this is correct. Weeping Willows is an indie rock group, thus they make rock music; Jane's Addiction is a rock band. Consequently the context confirms that the genre performed by both bands is 'rock'. The correct option is 'rock'. 
        """
        },
        
        {"role": "user", "content": """
        Question: Between two tennis players Kim Clijsters and Mary Pierce, who is older?
        Options: ['Kim Clijsters', 'Mary Pierce']
        Candidate answer: kim clijsters
        Suggestion: The correct answer is 'Mary Pierce' as she was born on 15 January 1975, which is earlier than Kim Clijsters who was born on 8 June 1983.
        Context: Kim Antonie Lode Clijsters (] ; born 8 June 1983) is a Belgian former professional tennis player. Clijsters is a former world No. 1 in both singles and doubles. Mary Pierce (born 15 January 1975) is a French retired tennis professional who played on the Women's Tennis Association (WTA) tour. Born in Canada, she is a citizen of Canada, and the United States. Pierce played for France in team competitions and in the Olympics.
        """
        },
        {"role": "assistant", "content": """
        Assistant: The candidate answer says that the older tennis player is kim clijsters, while the suggestion indicates mary pierce. The context provides the birth dates of both players, thus I can check which of the two options is correct. kim clijsters was born on 8 June 1983, and mary pierce was born on 15 January 1975. By comparing these dates, it's clear that mary pierce is older than kim clijsters. Thus the correct option is 'mary pierce'.
        """},
        {"role": "system", "content": "Now do the same for the following question:"},
        {"role": "user", "content": user_content}
    ]

    return messages

def preSynthGeneration(query, candidate_answer, critique, merged, sources):
    prompt = create_message_presynthesis(query, merged, candidate_answer, critique, sources)
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

#############################################

def synthesisGeneration(query, merged, pre_answer, sources):
    merged = ast.literal_eval(merged)
    augmented_prompt = synthesis_chain.invoke({'options': merged, 'suggestion': pre_answer})

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

#############################################

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="cuda",
    torch_dtype="auto",
    token = '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',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token = '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', use_fast=False)
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

##############################################################################################

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

##############################################################################################

# THESIS
answers = []
for i in range(N_rows):
    answers.append(extract_answer_thesis(thesisGeneration(first_queries[i], possibilities[i], sources[i])))


# ANTITHESIS
ant_answers = []
for i in range(N_rows):
    ant_answers.append(antithesisGeneration(first_queries[i], possibilities[i], answers[i], sources[i]))

# format antithesis
format_answers = []
for i in range(N_rows):
    format_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            ant_answers[i])))

# SYNTHESIS
pre_answers = []
for i in range(N_rows):
    pre_answers.append(preSynthGeneration(first_queries[i], possibilities[i], answers[i], ant_answers[i], sources[i]))


# format synthesis
syn_answers = []
for i in range(N_rows):
    syn_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            pre_answers[i])))

def_answers = ["The correct option is " + clean_text(correct_answer) + " due to what is said in the context." for correct_answer in correct_answers]

# format synthesis
oracle_answers = []
for i in range(N_rows):
    oracle_answers.append(extract_answer_synthesis(
        synthesisGeneration(
            first_queries[i], possibilities[i], 
            def_answers[i])))

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': ant_answers,
    'extracted antithesis': format_answers,
    'pre-synthesis': pre_answers,
    'synthesis': syn_answers,
    'oracle': oracle_answers,
    'context': sources
} 

df = pd.DataFrame(df)

# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)
df['synthesis'] = df['synthesis'].apply(clean_text_final)
df['oracle'] = df['oracle'].apply(clean_text_final)


##############################################################################################


df.to_csv('baseline-llama-3.1-instruct-8b-ctx.csv')
