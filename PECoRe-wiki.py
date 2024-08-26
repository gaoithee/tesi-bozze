import inseq
from inseq.commands.attribute_context.attribute_context import attribute_context_with_model, AttributeContextArgs
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import numpy as np
import pandas as pd
import torch
from transformers import BitsAndBytesConfig

##################################################################################

double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
model_double_quant = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", quantization_config=double_quant_config)

inseq_model = inseq.load_model(
    model_double_quant,
    "saliency",
)

df = pd.read_csv("wikihop_dataset/test-wikihop.csv")
df = df[:100]

tokenizer = AutoTokenizer.from_pretrained(model_double_quant) 

##################################################################################

# FUNZIONI AUSILIARIE VARIE

def top_k_positions(input_context_scores, k):
    # Convert the input array to a numpy array if it's not already
    input_context_scores = np.array(input_context_scores)
    
    # Use argsort to get the indices of the top 8 highest elements
    top_k_indices = np.argsort(input_context_scores)[-k:]
    
    # Reverse to get the indices in descending order of their values
    top_k_indices = top_k_indices[::-1]
    
    return top_k_indices.tolist()

# -------------------------------------------------------------------------

def estrai_frasi_con_token(frasipositioni, posizioni_token):
    # Creiamo un set per le posizioni dei token per una ricerca più veloce
    set_posizioni_token = set(posizioni_token)
    
    # Lista per memorizzare le frasi di contesto
    frasi_contesto = []
    
    for frase, inizio, fine in frasipositioni:
        # Verifica se c'è almeno un token rilevante nell'intervallo [inizio, fine]
        if any(token in set_posizioni_token for token in range(inizio, fine + 1)):
            frasi_contesto.append(frase)
    
    return frasi_contesto

# -------------------------------------------------------------------------

def find_positions(vector, target="<0x0A>"):
    positions = []
    for i, element in enumerate(vector):
        if element == target:
            positions.append(i)   
    return positions

# -------------------------------------------------------------------------

def map_relative_positions(abs_positions, text):
    temp = []
    start = 0
    for i in range(len(abs_positions)):
        if i != 0: 
            start = abs_positions[i-1] + 1
        temp.append([split_text(text)[i], start, abs_positions[i]])

    return temp

# -------------------------------------------------------------------------

def select_passages(text_passage, question, p, out, tokens):
    
    num_of_sets = len(out.cci_scores)
    
    top_scores = []
    for i in range(num_of_sets):
        top_scores.extend(top_k_positions(out.cci_scores[i].input_context_scores, p))
    positions = list(set(top_scores))

    return '\n'.join(estrai_frasi_con_token(map_relative_positions(find_positions(tokens), text_passage), positions))
 
# -------------------------------------------------------------------------

def split_text(text):
    # Split the text on ". " and return the resulting list
    return text.split("\n ")

# -------------------------------------------------------------------------

def invoke_pecore(text_passage, question, p):
    pecore_args = AttributeContextArgs(
        model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
        attribution_method="saliency",
        attributed_fn="contrast_prob_diff",
        context_sensitivity_metric="kl_divergence",
        context_sensitivity_std_threshold=1,
        context_sensitivity_topk = find_top_p(text_passage, p),
        attribution_std_threshold=None,
        attribution_topk=None,
        input_current_text=question, 
        input_context_text=text_passage,
        contextless_input_current_text="""<|system|>
    You are a helpful assistant that provide concise and accurate answers.<|end|>
    <|user|>
    {current}<|end|>
    <|assistant|>""",
        input_template="""<|system|>
    You are a helpful assistant that provide concise and accurate answers.<|end|>
    <|user|>
    {context}
    
    {current}<|end|>
    <|assistant|>""",
        contextless_output_current_text="""{current}""",
        output_template="{current}",
        special_tokens_to_keep=['<|system|>', '<|end|>', '<|assistant|>', '<|user|>'],
        decoder_input_output_separator="""
    """,
        save_path=None,
        viz_path=None,
        show_viz=False,
        generation_kwargs={'max_new_tokens': 50},
    )

    out = attribute_context_with_model(pecore_args, inseq_model) 
    return out

# -------------------------------------------------------------------------

def find_top_p(text_passage, p):
    
    tokens = tokenizer.tokenize(text_passage)
    num_topk_tokens = int(p/100 * len(tokens)) 

    return num_topk_tokens
    
##################################################################################

def run(question, passage, p):
    
    tokens = tokenizer.tokenize(passage)
    out = invoke_pecore(passage, question, p)
    return select_passages(passage, question, p, out, tokens)

selected_passages = []
for i in range(len(df)):
    selected_passages.append(run(df['query_x'][i], df['supports'][i], 5))
    
df['new'] = selected_passages

df.to_csv('ultramega-test-wikihop.csv')




