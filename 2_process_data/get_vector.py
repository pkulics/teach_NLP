#encoding=utf-8
from gensim.models import word2vec
import jieba

file_dir = "data/news.txt"
vec_dir = "data/vec.bin"
sentences = []
with open(file_dir,'r',encoding='utf-8') as corpus:
    for line in corpus:
        line = line.strip().strip("\n")
        line = jieba.lcut(line)
        sentences.append(line)
model = word2vec.Word2Vec(sentences, sg=1, size=10, window=5, min_count=1, negative=3,sample=0.001,hs=1,workers=4)
model.save(vec_dir)
print("首富:",model['首富'])

