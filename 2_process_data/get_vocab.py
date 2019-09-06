#coding=utf-8
import jieba

def is_chinese(uchar):
    if '\u4e00' <= uchar <= '\u9fff':
        return True
    else:
        return False

def get_stopwords(data_dir):
    with open(data_dir,'r',encoding='utf-8') as data:
        stopwords = set()
        for w in data:
            w = w.strip().strip('\n')
            stopwords.add(w)
        return stopwords

def get_vocab(data_dir,stopwords):
    with open(data_dir,'r',encoding='utf-8') as data:
        words = set()
        for line in data:
            line = line.strip().strip('\n')
            line = jieba.cut(line)
            for w in line:
                if is_chinese(w) and w not in stopwords :
                    words.add(w)
        return list(words)

if __name__ == '__main__':
    data_dir = "data/news.txt"
    stop_dir = "data/stopwords.txt"
    vocab_dir = "data/vocab.txt"
    stopwords = get_stopwords(stop_dir)
    words = get_vocab(data_dir,stopwords)
    with open(vocab_dir,'w',encoding='utf-8') as f:
        f.write("\n".join(words))
        
