#!/Users/okita/Applications/anaconda2/bin/python
# -*- coding: utf-8 -*-
import pickle
import MeCab

class MyDict():
    def __init__(self, filename):
        self.vocab = {}
        self.dictfile = filename
        self.mecab = MeCab.Tagger("-Owakati")
        
    def loadict(self):
        f = open(self.dictfile, "r")
        self.vocab = pickle.load(f)
        f.close()
        
    def savedict(self):
        f = open(self.dictfile, "w")
        pickle.dump(self.vocab, f)
        f.close()
        
    def word2id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word] 
    
    def words2ids(self, words):
        ids = []
        for word in words:
            ids.append(self.word2id(word))
        return ids;
    
    def ids2words(self, ids):
        words = []
        for id in ids:
            word = self.vocab.keys()[self.vocab.values().index(id)]
            words.append(word)
        return words;
    
    def lids2words(self, lids):
        words = []
        for ids in lids:
            word = self.ids2words(ids)
            words.append(word)
        return words
    
    
    def loadtext(self, filename):
        lines = open(filename).read().split('\n')
        ids = []
        for line in lines:
            words = line.split()
            id = self.words2ids(words)
            ids.append(id)
        return ids

    def countDictWords(self):
        return len(self.vocab)
    
    def wakachi(self, text):
        w = self.mecab.parse(text)
        return w.split()

if __name__ == '__main__':
    ## source activate root
    ## pip install prettyprint
    from prettyprint import pp
    dictfile = 'dict.dat'
    dict = MyDict(dictfile)
    
    #dict.loadict()
    lids = dict.loadtext('jp.txt')
    dict.savedict()
    
    #print lids
    #print dict.lids2words(lids)
    print dict.countDictWords()
    i = 0
    for ids in lids:
        jwords =  ','.join(dict.ids2words(ids))
        iwords = ','.join(map(str, ids))
        s = str(i) + '\t' + jwords + '\t' + iwords
        pp(s)
        i += 1
        
    
