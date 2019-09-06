#-*- coding: UTF-8 -*-

# 1 get the maxsentennum of sentence
def genmaxlength(data):
    maxsentencenum = 0
    for i in range(len(data)):
        if len(data[i]) > maxsentencenum:
            maxsentencenum = len(data[i])
    maxsentencenum = 56 #默认56
    return maxsentencenum

# 2 get the batchdata from data
def genBatch(data,maxsentencenum):
    doc = []
    for sentence in data:
        for i in range(maxsentencenum-len(sentence)):
            sentence.append(0)
        doc.append(sentence)
    return doc

class Dataset(object):
    def __init__(self,filename,emb,batch_size):
        Docs = map(lambda x:x.split(),open(filename).readlines())
        docW = []
        for sentence in Docs:
            SenW = []
            for word in sentence:
                word = word.lower()
                if word not in emb.voc:
                    # print word
                    word = -1
                word = emb.getID(word)
                SenW.append(word)
            docW.append(SenW)
        self.docsW = docW
        print self.docsW[0:10]
        self.docs = genBatch(docW,genmaxlength(docW))


class Wordlist(object):
    def __init__(self,filename,maxn = 100000-1):
        lines = map(lambda x:x.split(),open(filename).readlines()[:maxn])  # wordlist每一行（lines)对应一个单词

        self.size = len(lines) #100000
        self.voc = [(item[0][0],item[1]+1) for item in zip(lines,xrange(self.size))]

        print(type(self.voc))
        self.voc = dict(self.voc) # 转换成了字典（{'is.it'：59613,'baseheart'：59614, 'beecham'：59615,'diverged'：59616, '1080p'： 59617})


    def getID(self,word):
        try:
            return self.voc[word]
        except:
            return 0

class Label(object):
    def __init__(self,filename):
        with open(filename, 'r') as file1:
            label = []
            for line in file1.readlines():
                label.append(int(line))
        self.label = label

