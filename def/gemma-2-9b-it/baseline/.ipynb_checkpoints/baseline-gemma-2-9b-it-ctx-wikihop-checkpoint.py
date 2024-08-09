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

def create_message_antithesis_ctx(question, candidate, options, context):
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
        Question: What languages did John Osteen speak or write?
        Options: ['english', 'greek', 'koine greek', 'nahuatl', 'spanish']
        Candidate answer: english
        Context: Christianity is a monotheistic religion based on the life and teachings of Jesus Christ. It is the largest religion in the world, with over 2 billion followers. Christians believe in one God and follow the teachings of Jesus Christ, who they believe is the Son of God and the savior of humanity.\n\nLakewood Church is a nondenominational charismatic Christian megachurch located in Houston, Texas. It is the largest congregation in the United States, with an average attendance of about 52,000 people per week. The church is led by Joel Osteen, who is the son of John Hillery Osteen, the founder and first pastor of Lakewood Church.\n\nMexico is a federal republic in North America, bordered by the United States, Guatemala, Belize, the Caribbean Sea, and the Pacific Ocean. It is the most populous Spanish-speaking country in the world and the second most populous country in Latin America. Mexico is a federation comprising 31 states and a federal district that is also its capital and most populous city.\n\nJohn Hillery Osteen founded Lakewood Church in Houston, Texas, in 1959. He served as the first pastor until his death in 1999. His son, Joel Osteen, succeeded him as pastor and continues to lead the church today.\n\nHouston is the most populous city in Texas and the fourth-most populous city in the United States. It is located in Southeast Texas near the Gulf of Mexico and is the seat of Harris County. Houston is the principal city of the Houston metropolitan area, which is the fifth-most populated metropolitan area in the United States.\n\nIn summary, Christianity is a monotheistic religion based on the life and teachings of Jesus Christ, with over 2 billion followers worldwide. Lakewood Church in Houston, Texas, is the largest congregation in the United States, led by Joel Osteen, the son of John Hillery Osteen, the founder and first pastor of the church. Houston is the most populous city in Texas and the fourth-most populous city in the United States, and is the principal city of the Houston metropolitan area.
        
        Assistant: 
        """
        },
        {"role": "assistant", "content": """
        The context states that John Hillery Osteen is the founder and first pastor of Lakewood Church in Houston, Texas. Even though Mexico is the most populous Spanish-speaking country in the world and the second most populous country in Latin America, we do not have evidence that the language is spanish. 
        The options 'greek', 'koine greek', 'nahuatl' do not figure in the context and are not proper answers. Thus the most reasonable option is 'english'.
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
    prompt = create_message_antithesis_ctx(query, candidate, merged, sources)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=500)
    return extract_antithesis(tokenizer.decode(outputs[0]))

#############################################

def create_message_presynthesis_ctx(question, candidate, suggestion, options, context):
    user_content = "Question: " + question + "\n Options: " + str(options) + "\n Candidate answer: " + candidate + "\n Suggestion: " + suggestion + "\n Context: " + context 

    chat = [
        {"role": "user", "content": """
        You are an helpful AI assistant. You are asked to determine the most correct answer for a given question, provided a set of possible options.
        You also have at disposal a first tentative answer and a suggestion on which is the correct answer.
        Your goal is to decree which is the most correct answer to the question between the available options according to the context.

        Here's a few examples on how to do it:
        Question: What languages did John Osteen speak or write?
        Options: ['english', 'greek', 'koine greek', 'nahuatl', 'spanish']
        Candidate answer: english
        Suggestion:  The context states that John Hillery Osteen is the founder and first pastor of Lakewood Church in Houston, Texas. Even though Mexico is the most populous Spanish-speaking country in the world and the second most populous country in Latin America, we do not have evidence that the language is spanish. The options 'greek', 'koine greek', 'nahuatl' do not figure in the context and are not proper answers. Thus the most reasonable option is 'english'.
        Context: Christianity is a monotheistic religion based on the life and teachings of Jesus Christ. It is the largest religion in the world, with over 2 billion followers. Christians believe in one God and follow the teachings of Jesus Christ, who they believe is the Son of God and the savior of humanity.\n\nLakewood Church is a nondenominational charismatic Christian megachurch located in Houston, Texas. It is the largest congregation in the United States, with an average attendance of about 52,000 people per week. The church is led by Joel Osteen, who is the son of John Hillery Osteen, the founder and first pastor of Lakewood Church.\n\nMexico is a federal republic in North America, bordered by the United States, Guatemala, Belize, the Caribbean Sea, and the Pacific Ocean. It is the most populous Spanish-speaking country in the world and the second most populous country in Latin America. Mexico is a federation comprising 31 states and a federal district that is also its capital and most populous city.\n\nJohn Hillery Osteen founded Lakewood Church in Houston, Texas, in 1959. He served as the first pastor until his death in 1999. His son, Joel Osteen, succeeded him as pastor and continues to lead the church today.\n\nHouston is the most populous city in Texas and the fourth-most populous city in the United States. It is located in Southeast Texas near the Gulf of Mexico and is the seat of Harris County. Houston is the principal city of the Houston metropolitan area, which is the fifth-most populated metropolitan area in the United States.\n\nIn summary, Christianity is a monotheistic religion based on the life and teachings of Jesus Christ, with over 2 billion followers worldwide. Lakewood Church in Houston, Texas, is the largest congregation in the United States, led by Joel Osteen, the son of John Hillery Osteen, the founder and first pastor of the church. Houston is the most populous city in Texas and the fourth-most populous city in the United States, and is the principal city of the Houston metropolitan area.
        """
        },
        {"role": "assistant", "content": """
        Assistant: Both the candidate answer and the suggestion agree on the fact that the correct option is 'english'. Let's check on the context whether or not this is correct. 'greek', 'koine greek', 'nahuatl' are out of topic, 'spanish' is not mentioned in the context as a possible language spoken by John Osteen. What is sure instead is that the english church in Mexico has english-speakers. Therefore, the correct option is 'english'.
        """
        },
        {"role": "user", "content": """
        Question: Where did Jim Wither Spoon receive his education?
        Options: ['new york', 'ohio state university', 'today']
        Candidate answer: today
        Suggestion: 'today' is not the correct option, since the question was about the place in which Jim Wither Spoon received his education, and 'today' is a temporal reference instead. 
        Context: Los Angeles, known as the City of Angels, is the second-most populous city in the United States, with a diverse population and a significant role in the entertainment industry. The city is surrounded by mountains and has a Mediterranean climate. Ice hockey is a popular sport played on ice, with two teams of six players each, aiming to score goals by shooting a vulcanized rubber puck into the opponent's net. The goaltender, positioned in front of the net, plays a crucial role in preventing goals and is protected by special equipment.\n\nThe Ohio State Buckeyes represent The Ohio State University in various sports, including ice hockey, which competes in the Western Collegiate Hockey Association (WCHA). The team's mascot, Brutus Buckeye, is a student dressed in Buckeye colors with a headpiece resembling an Ohio Buckeye nut.\n\nJames Witherspoon, a retired Canadian professional ice hockey defenceman, played for the Ohio State Buckeyes and briefly for the Los Angeles Kings in the National Hockey League (NHL). The Los Angeles Kings are a professional ice hockey team based in Los Angeles, a member of the Pacific Division of the Western Conference of the NHL. The team was founded in 1967 and moved to the Staples Center in Downtown Los Angeles in 19992000.\n\nNew York City, the most populous city in the United States, is a global power city with a significant impact on commerce, finance, media, art, fashion, research, technology, education, and entertainment. It is the center of the New York metropolitan area and home to the United Nations headquarters. The National Hockey League (NHL) is a professional ice hockey league with 31 member clubs, considered the premier professional ice hockey league in the world. The Stanley Cup, the oldest professional sports trophy in North America, is awarded annually to the league playoff champion.
        """
        },
        {"role": "assistant", "content": """
        Assistant: Jim Wither Spoon, referred to in the text as James Witherspoon, played for the Ohio State Buckeyes. The context clearly states that 'the Ohio State Buckeyes represent The Ohio State University in various sports, including ice hockey', On the opposite, there's no mention of him being associated with 'new york' in terms of education. New York City is mentioned in the text in a different context, related to its status and impact, but not connected to Witherspoon's education. Therefore, the correct option is 'ohio state university'.
        """},
        {"role": "user", "content": "Now do the same for the following question: "+ user_content}
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def preSynthGeneration(query, candidate_answer, critique, merged, sources):
    prompt = create_message_presynthesis_ctx(query, merged, candidate_answer, critique, sources)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=500)
    return extract_antithesis(tokenizer.decode(outputs[0]))

#############################################

def synthesisGeneration(query, merged, pre_answer):
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

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast = False)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

new_model = models.Transformers(model, tokenizer, temperature=0.0)

#############################################

df = pd.read_csv('wikihop_dataset/wikihop-merged-summarized-test.csv')

# select a subset of the queries, just for test:
first_queries = df['query']

# same for correct answers and distractors:
correct_answers = df['answer']
possibilities = df['options']

# and for the sources:
sources = df['sum_supports']

N_rows = len(df)

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

#############################################

df = {
    'query': first_queries,
    'correct': correct_answers,
    'thesis': answers,
    'antithesis': ant_answers,
    'pre-synthesis': pre_answers,
    'context': sources
} 

df = pd.DataFrame(df)


# Applica la funzione alla colonna 'correct answer'
df['correct'] = df['correct'].apply(clean_text_final)
df['thesis'] = df['thesis'].apply(clean_text_final)

df.to_csv('baseline-ctx-gemma-2-9b-it-wikihop.csv')


