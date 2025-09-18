# Simpsons Word2Vec

**Description:** Word2Vec embeddings trained on Simpsons scripts for NLP experiments.

## Project Overview
This project demonstrates training **Word2Vec models (CBOW & Skip-gram)** on Simpsons TV show scripts. It includes:
- Data preprocessing with `spaCy`
- Text cleaning, tokenization, and lemmatization
- Training Word2Vec embeddings with Gensim
- Examples of similar words using trained embeddings

## Files
- `preprocess.py` : Preprocess Simpsons scripts and generate tokens
- `word2vectrain.py` : Train Word2Vec model (CBOW or Skip-gram)
- `tokens_processed.json` : Preprocessed tokenized text
- `requirements.txt` : Required Python libraries

## Installation
```bash
git clone https://github.com/marwaMahmoud/simpsons-word2vec.git
cd simpsons-word2vec
pip install -r requirements.txt


## Usage

Preprocess the scripts:
python preprocess.py

## Train Word2Vec:

python word2vectrain.py


## Test embeddings:

from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_cbow.model")
print(model.wv.most_similar("child", topn=10))
