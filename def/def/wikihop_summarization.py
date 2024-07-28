import pandas as pd
from transformers import pipeline
import re
import random
from guidance import models, select
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('wikihop-dataset-1000.csv')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn",)
#print(df['supports'][100])

print('Start summarizing...')

summarizations = []
for i in range(len(df)):
    summarizations.append(summarizer(df['supports'][i], max_length=400, min_length=10, do_sample=False))

print('Done!')

print(summarizations)

df['summarization'] = summarizations

df.to_csv('wikihop-summarization.csv')