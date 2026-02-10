import pandas as pd
import json


from keybert import KeyBERT
kw_model = KeyBERT()

df_train = pd.read_csv('train.csv')

df_train = df_train[['utterances']]

words = []

for i in df_train['utterances'].to_list():
    for j in i.split(' '):
        if j not in words:
            words.append(j)

def get_keyBert_keywords(text):
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')

def return_keywords(text):
    text_keyword = get_keyBert_keywords(text)
    final_keyword = []
    for keyword in text_keyword:
        final_keyword.append(keyword[0])
    
    final_keyword = str(' '.join(final_keyword))
    return final_keyword

with open('concept/cc_kg.json', encoding='utf-8') as file:
    data = json.load(file)

text = 'Hey man , you wanna buy some weed ?'
keys = return_keywords(text)

# print(return_keywords(text))
triples = []

for key in keys.split(' '):
    for x in data[key]:
        if x[1] in ['FormOf', 'IsA', 'Causes', 'Synonym', 'SimilarTo', 'HasA'] and x[2] in words:
            triples.append(' '.join(x))

for x in triples:
    print(x)

