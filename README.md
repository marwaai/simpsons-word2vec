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
```
## text_pre_processing
text_pre_processing.py
## Train Word2Vec
 word2vectrain.py.py
## Test embeddings
test.py
