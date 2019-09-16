#encoding:utf-8
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np

vec_dir = "data/vec.bin"
vocab_dir = "data/vocab.txt"
select_vocab = "data/select_vocab.txt"

vector = []
model = Word2Vec.load(vec_dir)

with open(vocab_dir,'r',encoding='utf-8') as vocab:
    for line in vocab:
        vec = ""
        line = line.strip('\n')
        try:
            # 在模型中查找
            vec = model[line]
        except:
            # 如果没有则随机初始化
            vec = list(np.random.uniform(-0.25,0.25,300))
        vector.append(" ".join(map(str,vec)))
# 保存
with open(select_vocab,'w',encoding='utf-8') as f:
    f.write("\n".join(vector))
    
