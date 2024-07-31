import pandas as pd
import torch
import re
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

import warnings
warnings.filterwarnings("ignore")

dataset = load_dataset('saracandu/my_wikihop', split='train')
sources = dataset['supports']

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summ = []
for i in range(len(sources)):
  if len(sources[i]) < 4000:
    summ.append(summarizer(sources[i], max_length=500, min_length=10, do_sample=False))
  else:
    parts = ''
    for j in range(int(len(sources[i])/4000)):
      temp = summarizer(sources[i][j*4000:(j+1)*4000], max_length = 100, min_length = 10, do_sample = False)
      parts = parts + temp[0]['summary_text']
    summ.append(parts)

df['summary'] = summ

df.to_csv('summarize-wikihop.csv')