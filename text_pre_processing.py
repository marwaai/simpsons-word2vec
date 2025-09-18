import json
import re
import spacy
from tqdm import tqdm 
from spacy.lang.en.stop_words import STOP_WORDS

data = []
with open("simpsons_lines.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))  # يحول كل سطر dict
ninp=[]
for i in data:
    ninp.append(i["out"])
ninp=" ".join(ninp)
cinp = (re.sub("[^A-Za-z.']+", ' ',ninp.lower()))
finp=[]
cinp=cinp.split()

del ninp
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
txt = [
    token.lemma_
    for doc in nlp.pipe(cinp, batch_size=5000)
    for token in doc
    if token.lemma_ not in STOP_WORDS and len(token.lemma_) > 2
]
del cinp 
with open("tokens_processed.json", "w", encoding="utf-8") as f:
    json.dump(txt, f, ensure_ascii=False, indent=2)