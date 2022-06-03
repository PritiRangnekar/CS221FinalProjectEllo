import csv
import pandas as pd
from summarizer import Summarizer

# Code for generating commentary using the BERT Extractive summarizer. 
df = pd.read_csv('./CONTEXT_b1Val')
# print(df)
texts = df['Text']
summaries = []

for text in texts: 
    model = Summarizer() 
    result = model(text, min_length = 2, ratio=0.2)
    summaries.append(result)

results = pd.DataFrame({'Text': texts, 'Summary': summaries, 'ISBN': df['ISBN'], 'Page': df['Page'] })
results.to_csv('summaries_contextb1Val.csv', index=False)


