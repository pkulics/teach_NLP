
from gensim.models import word2vec

#file_dir = "pingjia/train.txt"
#vec_dir = "pingjia/vec.bin"
file_dir = "yt/train.txt"
vec_dir = "yt/vec.bin"
sentences = []
with open(file_dir,'r',encoding='utf-8') as corpus:
    for line in corpus:
        line = line.split('\t')[1]
        line = line.split()
        sentences.append(line)
#sentences = word2vec.Text8corpus(file_dir)
model = word2vec.Word2Vec(sentences, sg=1, size=300, window=5, min_count=5, negative=3,sample=0.001,hs=1,workers=4)
model.save(vec_dir)
print(model['贵'])

