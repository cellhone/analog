#encoding: utf-8
"""
Desc:   
Author: Makoto OKITA
Date:   2016/09/03   
"""
import numpy as np
import sys
import json
import itertools
import os, ConfigParser
import pickle

from prettyprint import pp
from mydict import MyDict
from rnn_text import Trainer
from rnn_analize import AnazlizeTrainer

def fload(filename):
    f = open(filename, "r")
    data = pickle.load(f)
    f.close()
    return data

def fsave(filename, data):
    f = open(filename, "w")
    pickle.dump(data, f)
    f.close()
    return


## Test Main
if __name__ == "__main__":
    dictfile = 'dict.dat'
    configFile = 'analog.ini'
    acceptFile = 'accept.dat'   ## 採用するメッセージid一覧
    dict = MyDict(dictfile)
    dict.loadict()
    print 'dict count: %d' % dict.countDictWords()
    
    ## RNNメッセージID化
    vocab=10000
    dim=100
    y=94
    train = Trainer(vocab, dim, y)
    train.load('train')
    
    labellist = []  ## ラベル一覧(出力)
    #msglist = []    ## メッセージ一覧（無視リスト作成のため使用メッセージを保存）
    analizeRnnInputList = []
    analizeRnnLabelList = []
    
    arg1 = sys.argv[1] if len(sys.argv) == 2 else None
    ## 学習 ##
    if arg1 == 'train':
        print "training...parsing..."
        ## エラー定義の読み込み
        with open('error.json', 'r') as f:
            errjson = json.load(f, "utf-8")
        for erritem in errjson:
            labellist.append(erritem)
            msgidlist = []
            for msgtxt in erritem['message']:
                ## wakachiで処理するため明示的にunicodeにする
                ## ex) cmapeerdが停止したためプロセスを再起動します
                w = u'' + msgtxt
                words = dict.wakachi(w.encode('utf-8'))
                ids = dict.words2ids(words)
                ## ex) [113, 63, 11, 9, 13, 104, 6, 7, 66, 8, 9, 10]
                #y = train.predict(ids)
                y = train.predict(ids[::-1])
                wordid = y.data.argmax(1)[0]    ## ex) 84
                msgidlist.append(wordid)
                
            analizeRnnInputList.append(msgidlist)
            pp(erritem)
            pp(msgidlist)
        ## ex) [[65,66],[40,86,84]]
        pp(analizeRnnInputList)
        
        ## 採用メッセージリスト保存
        ## [[83], [65, 66]...] こうなっているのを [83,65,66...]にしてユーニークにする
        fsave(acceptFile, list(set([flatten for inner in analizeRnnInputList for flatten in inner])))
        
        print "training...learning"
        trainData = analizeRnnInputList
        
        
        config = ConfigParser.SafeConfigParser()
        config.read(configFile)
        
        dim_in  = config.getint('analize', 'dim_in') ## 全パラメータ数？
        dim_mid = config.getint('analize', 'dim_mid')
        dim_out = len(trainData)
        epoch = config.getint('analize', 'epoch')
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        for j in range(epoch):
            i = 0
            for ids in trainData:
                #pp(ids)
                ## 全パターン学習するか
                if False:
                    for l in itertools.permutations(ids):
                        x = list(l)
                        #print(x)
                        #loss, y = train.practice(x[::-1], i)
                        loss, y = train.practice(x, i)
                else:
                    loss, y = train.practice(ids[::-1], i)
                #print loss.data
                i += 1
            #if j % 10 == 0:
            #    print loss.data
            print loss.data
        train.save('train_analize')

        config.set('analize', 'dim_in', str(dim_in))
        config.set('analize', 'dim_mid', str(dim_mid))
        config.set('analize', 'dim_out', str(dim_out))
        config.write(open(configFile, 'w'))

            
    ## 予測 ##
    elif arg1 == 'predict':
        print 'predict...'
        
        """
        inmsg = [
            'cmapeerdが停止したためプロセスを再起動します', 
            'cmapeerdの自動再起動に失敗しました'
        ]
        """
        #inmsg = ['エージェントが可能な変更を行った後動作を開始しました']
        
        """
        inmsg = [
            "クラスターサービスが停止しました(disabled)",
            "クラスターサービスを実行しているサーバーが変更されました",
            "クラスターサービスの状態がstarted以外に変更されました",
            "クラスターのメンバー数が減りました"
        ]
        """
        
        inmsg = [
            'cmapeerdが停止したためプロセスを再起動します', 
            'cmapeerdの自動再起動に失敗しました',
            "クラスターサービスが停止しました(disabled)",
            "クラスターサービスを実行しているサーバーが変更されました",
            "クラスターサービスの状態がstarted以外に変更されました",
            "クラスターのメンバー数が減りました"

            
        ]
        
        

        analizeRnnInputList = []

        msgidlist = []
        for msgtxt in inmsg:
            ## wakachiで処理するため明示的にunicodeにする
            ## ex) cmapeerdが停止したためプロセスを再起動します
            #w = u'' + msgtxt
            #words = dict.wakachi(w.encode('utf-8'))
            words = dict.wakachi(msgtxt)
            ids = dict.words2ids(words)
            ## ex) [113, 63, 11, 9, 13, 104, 6, 7, 66, 8, 9, 10]
            y = train.predict(ids[::-1])
            wordid = y.data.argmax(1)[0]    ## ex) 84
            msgidlist.append(wordid)

        pp(inmsg)
        pp(msgidlist)
        
        
        
        config = ConfigParser.SafeConfigParser()
        config.read(configFile)
        
        dim_in  = config.getint('analize', 'dim_in')
        dim_mid = config.getint('analize', 'dim_mid')
        dim_out = config.getint('analize', 'dim_out')
        
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        train.load('train_analize')
        y = train.predict(msgidlist[::-1])
        print y.data.argmax(1)[0]
        rank = y.data.argsort()[0]
        uprank = map(int, rank[::-1])
        print uprank
        #print y.data[0]

        for i in uprank:
            print '%d, %2f' % (i, y.data[0][i])
        print ''