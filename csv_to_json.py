import re
import pandas as pd
import spacy 
import logging  # Setting up the loggings to monitor gensim
from collections import defaultdict  # For word frequency
from time import time  # To time our operations
from tqdm import tqdm
import json
#raw_texts
df0=pd.read_csv(r"C:\Users\marwa\symsom\text classfier symson\simpsons_script_lines.csv",chunksize=1000)
df = pd.concat([chunk for chunk in tqdm(df0, desc="Reading CSV")])
print(df["raw_text"].isnull().sum())
inp=[]
out=[]
for i in tqdm(df["raw_text"]):
      inp.append(i.split(":")[0])
      out.append(i.split(":")[1])

with open("simpsons_lines.jsonl", "w", encoding="utf-8") as f:
    for inp, out in zip(inp, out):
        json.dump({"inp": inp, "out": out}, f, ensure_ascii=False)
        f.write("\n")


