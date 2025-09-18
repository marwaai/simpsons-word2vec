import multiprocessing
import time
from gensim.models import Word2Vec
import json

tokens=[]
tokens= json.load(open('tokens_processed.json', 'r', encoding='utf-8'))
data = [tokens[i:i+20] for i in range(0, len(tokens), 20)]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=2,
                     window=2,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1,   compute_loss=True,sg=0 
                     #for cbow sg=0 for skipgram=1 (i train two model one with cbow and another with skipgram)
)
t = time.time()
w2v_model.build_vocab(data, progress_per=100)

print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

previous_loss = 0
epochs = 15

for epoch in range(epochs):
    w2v_model.train(
        data,
        total_examples=w2v_model.corpus_count,
        epochs=1
    )
    current_loss = w2v_model.get_latest_training_loss()
    epoch_loss = current_loss - previous_loss
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
    previous_loss = current_loss
w2v_model.save("word2vec_cbow.model")
