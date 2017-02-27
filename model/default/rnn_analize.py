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
import itertools


"""
基本処理
"""
class RnnAnalize(Chain):
    def __init__(self, v, k, y):
        super(RnnAnalize, self).__init__(
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
class AnazlizeTrainer():
    def __init__(self, v, k, y):
        self.model = RnnAnalize(v, k, y)
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

        
## Test Main
if __name__ == "__main__":
    import sys
    import io
    import re
    arg1 = sys.argv[1] if len(sys.argv) == 2 else None 
    
    trainData = [[4], [1,2,3], [10,11,12], [1,22,23], [1], [5],[6],[7],[8],[9] ]
    #for data in baseData:
    #    for i in itertools.permutations(data):
    #        trainData.append( list(i) )
    print trainData
    print len(trainData)
    
    
    #dim_in  = 1000
    #dim_mid = 100
    #dim_out = len(trainData)
    dim_in  = len(trainData)
    dim_mid = 50
    dim_out = len(trainData)
    epoch = 1
    
    ## 学習
    if arg1 == 'train':
        print "training..."
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        for j in range(epoch):
            i = 0
            for ids in trainData:
                #pp(ids)
                if True:
                    for l in itertools.permutations(ids):
                        x = list(l)
                        #print(x)
                        #loss, y = train.practice(x[::-1], i)
                        loss, y = train.practice(x, i)
                else:
                    loss, y = train.practice(ids[::-1], i)
                    #loss, y = train.practice(ids, i)
                #print loss.data
                i += 1
            #if j % 10 == 0:
            #    print loss.data
            print loss.data
            train.save('train_analize')
            

    
    ## 予測
    elif arg1 == 'predict':
        print 'predict...'
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        train.load('train_analize')
        while True: 
            #train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
            #train.load('train_analize')
            
            ids = map(int, raw_input().split())
            print ids
            y = train.predict(ids[::-1])
            print y.data.argmax(1)[0]
            rank = y.data.argsort()[0]
            uprank = map(int, rank[::-1])
            print uprank
            #print y.data[0]

            for i in uprank:
                print '%d, %2f' % (i, y.data[0][i])
            print ''