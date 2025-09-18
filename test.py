from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_cbow.model")
similar_words = model.wv.most_similar("child", topn=10)
print("cw",similar_words)
model = Word2Vec.load("word2vec.model")
similar_words = model.wv.most_similar("child", topn=10)
print("sk:",similar_words)