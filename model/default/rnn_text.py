#encoding: utf-8
"""
Desc:   
Author: Makoto OKITA
Date:   2016/09/03   
"""
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.preprocessing import StandardScaler    ## 標準化




"""
基本処理
"""
class RnnChain(Chain):
    def __init__(self, v, k, y):
        super(RnnChain, self).__init__(
            embed = L.EmbedID(v, k),
            H  = L.LSTM(k, k),
            W = L.Linear(k, y),
        )


    def __call__(self, x, y):
        accum_loss = None
        v, k = self.embed.W.data.shape
        self.H.reset_state()                 
        for i in range(len(x)):
            nx = Variable(np.array([x[i]], dtype=np.int32))
            ny = Variable(np.array([y], dtype=np.int32))
            
            wx = self.embed(nx)
            wh = self.H(wx)
            ww = self.W(wh)
            
            loss = F.softmax_cross_entropy(ww, ny)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss, ww

    def forward(self, x):
        for i in range(len(x)):
            nx = Variable(np.array([x[i]], dtype=np.int32))
            
            wx = self.embed(nx)
            wh = self.H(wx)
            ww = self.W(wh)
            
        return ww
    
    
"""
学習・予測処理
"""
class Trainer():
    def __init__(self, v, k, y):
        self.model = RnnChain(v, k, y)
        #self.model.compute_accuracy = False  # accuracyが必要ない場合はFalseした方が学習が速い？
        self.optimizer = optimizers.Adam() # Adam, AdaGrad, AdaDelta, RMSpropGraves, SGD, MomentumSGD
        self.optimizer.setup(self.model)
        #self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))  #??? 荷重減衰による正則化 ??? saveで保存されない！？

        
    ### 学習
    def practice(self, x, y):
        self.model.H.reset_state()
        self.model.zerograds()        
        loss, y = self.model(x, y)
        loss.backward()
        #loss.unchain_backward()  # truncate        
        self.optimizer.update()
        return loss, y
    
    
    ### 予測
    def predict(self, x):
        self.model.H.reset_state()
        self.model.zerograds() 
        y = self.model.forward(x)
        return F.softmax(y)
    
    
    def save(self, filename):
        #modelとoptimizerを保存
        serializers.save_npz(filename +'_model.dat', self.model)
        serializers.save_npz(filename +'_optimizer.dat', self.optimizer)
        
        
    def load(self, filename):
        serializers.load_npz(filename +'_model.dat', self.model)
        serializers.load_npz(filename +'_optimizer.dat', self.optimizer)


##########
## 標準化
class Standardization():
    def __init__(self):
        self.ss = StandardScaler()
        #self.fit_value
        #self.fit_obj
        
    def fit(self, data):
        self.fit_obj = self.ss.fit(data)
        return self.fit_obj.transform(data)
        
    def transform(self, data):
        return self.fit_obj.transform(data)
    
        
## Test Main
if __name__ == "__main__":
    from prettyprint import pp
    from mydict import MyDict
    import sys
    
    arg1 = sys.argv[1] if len(sys.argv) == 2 else None 
    ## 学習
    if arg1 == 'train':
        print "training..."

        dictfile = 'dict.dat'
        dict = MyDict(dictfile)
        dict.loadict()
        lids = dict.loadtext('jp.txt')
        dict.savedict()


        #vocab = dict.countDictWords()
        vocab = 10000
        dim = 100
        y = len(lids)
        print'vocab=%d, dim=%d, y=%d' % (vocab, dim, y)
        train = Trainer(vocab, dim, y)

        epoch = 1000
        for j in range(epoch):
            i = 0
            for ids in lids:
                #pp(ids)
                #pp(i)
                loss, y = train.practice(ids[::-1], i)
                #print loss.data
                i += 1
            #if j % 10 == 0:
            #    print loss.data
            print loss.data
            train.save('train')
    
    ## 予測
    elif arg1 == 'predict':
        print 'predict...'
        ## 12,"システム,管理,プロセス,を,起動,し,ます","35,5,6,7,8,9,10"
        #ids = [35,5,6,7,8,9,10]
        dictfile = 'dict.dat'
        dict = MyDict(dictfile)
        dict.loadict()
        

        #text = 'システム管理プロセスを起動します'
        #text = 'クラスターサービスの状態がstarted以外に変更されました'
        text = 'エージェントが可能な変更を行った後動作を開始しました'
        words = dict.wakachi(text)
        ids = dict.words2ids(words)
        pp(words)
        print ids
            
        vocab=10000
        dim=100
        y=94
        train = Trainer(vocab, dim, y)
        train.load('train')
        print ids
        y = train.predict(ids[::-1])
        print y.data.argmax(1)[0]
        rank = y.data.argsort()[0]
        uprank = map(int, rank[::-1])
        print uprank
        print y.data[0]
        
        for i in uprank:
            print '%d, %2f' % (i, y.data[0][i])
            