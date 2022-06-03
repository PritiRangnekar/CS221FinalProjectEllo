import csv
import pandas as pd
from summarizer import Summarizer

df = pd.read_csv('./CONTEXT_b1Val')
# print(df)
texts = df['Text']
summaries = []

for text in texts: 
    # print("in for loop")
    # print("text: ", text)
    model = Summarizer() 
    result = model(text, min_length = 2, ratio=0.2)
    #full = ''.join(result)
    # print(result)
    summaries.append(result)
    # print(summaries)
    # print("end of for loop")

results = pd.DataFrame({'Text': texts, 'Summary': summaries, 'ISBN': df['ISBN'], 'Page': df['Page'] })
results.to_csv('summaries_contextb1Val.csv', index=False)


