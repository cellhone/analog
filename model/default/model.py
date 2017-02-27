#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-
"""
Desc:   AnaLog Model file
Author: Makoto OKITA
Date:   2016/10/26       
"""
import numpy as np
import os, ConfigParser
import csv, copy
import traceback, logging
import json
import pickle

from prettyprint import pp
from mydict import MyDict
from rnn_text import Trainer
from rnn_analize import AnazlizeTrainer


## ログ
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
log.setLevel(logging.ERROR)
log.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


## Model Class
class AnalogModel:
    def __init__(self):
        self.model_path = os.path.dirname(__file__) ## modelのPATH
        self.config_file = 'model.ini'  ## model設定ファイル
        self.loadConfig(self.model_path + '/' + self.config_file)

        
    ## 設定値読み込み
    def loadConfig(self, file):
        config = ConfigParser.SafeConfigParser()
        config.read(file)
        self.modelname = config.get('MODEL', 'model_name')
        self.traindata_file = config.get('TRAIN', 'data_file')
        self.trainlabel_file = config.get('TRAIN', 'label_file')
        self.epoch = config.getint('TRAIN', 'epoch')
        self.log_level = getattr(logging, config.get('MODEL','log_level'))    ## LOG LEVEL!!
        log.setLevel(self.log_level)

    
    ## csv読み込み、2次元配列 [[0,0],[1,1]]
    def loadCSV(self, file):
        #data = [[ int(elm) for elm in v] for v in csv.reader(open(file, "r"))]
        csv_obj = csv.reader(open(file, 'r'), delimiter=',')    ## 文字列として読み込まれる
        data = [[int(elm) for elm in v] for v in csv_obj]   ## 数値型に変換
        return data
    
    
    def getTraindata(self):
        ## オリジナルが変更されないようにdeep copy
        return copy.deepcopy(self.traindata)
    
    
    def fload(self, filename):
        f = open(filename, "r")
        data = pickle.load(f)
        f.close()
        return data
    

    ## 予測
    def predict(self, data):
        inmsg = data.split('\n')
        #inmsg = ['エージェントが可能な変更を行った後動作を開始しました']

        log.debug('predict: predict　data=%s', inmsg)
        dictfile = self.model_path + '/dict.dat'
        configFile = self.model_path + '/analog.ini'
        dict = MyDict(dictfile)
        dict.loadict()
        print 'dict count: %d' % dict.countDictWords()
        
        ## 採用メッセージID
        acceptid = self.fload(self.model_path + '/accept.dat')
        print '**accept id:'
        print acceptid
        
        ## エラー定義の読み込み
        with open(self.model_path + '/error.json', 'r') as f:
            errjson = json.load(f, "utf-8")

        ## RNNメッセージID化
        vocab=10000
        dim=100
        y=94
        train = Trainer(vocab, dim, y)
        train.load(self.model_path + '/train')

        labellist = []  ## ラベル一覧(出力)
        #msglist = []    ## メッセージ一覧（無視リスト作成のため使用メッセージを保存）
        analizeRnnInputList = []
        analizeRnnLabelList = []
        msgidlist = []
        for msgtxt in inmsg:
            ## wakachiで処理するため明示的にunicodeにする
            ## ex) cmapeerdが停止したためプロセスを再起動します
            #w = u'' + msgtxt
            #words = dict.wakachi(w.encode('utf-8'))
            words = dict.wakachi(msgtxt)
            ids = dict.words2ids(words)
            ## 改行のみはスキップ
            if ids == []:
                continue
            ## ex) [113, 63, 11, 9, 13, 104, 6, 7, 66, 8, 9, 10]
            y = train.predict(ids[::-1])
            wordid = y.data.argmax(1)[0]    ## ex) 84
            
            ## エラーリストにあるメッセージ以外は処理にしない
            if wordid in acceptid:
                msgidlist.append(wordid)
            else:
                print 'not accept id:%d, msg:%s' % (wordid, msgtxt)
                continue
            print '** msg prdict %d, %2f' % (wordid, y.data[0][wordid]) ## 選択されたidの確率表示
        pp(inmsg)
        pp(msgidlist)
        
        ## 対象メッセージが無い
        if msgidlist == []:
            result = [{
                'score': 100,
                'id': None,
                'label': None
            }]
            return result
            
        
        ## エラーケースの予測
        config = ConfigParser.SafeConfigParser()
        config.read(configFile)
        
        dim_in  = config.getint('analize', 'dim_in')
        dim_mid = config.getint('analize', 'dim_mid')
        dim_out = config.getint('analize', 'dim_out')
        
        """
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        train.load(self.model_path + '/train_analize')
        y = train.predict(msgidlist[::-1])
        print y.data.argmax(1)[0]
        rank = y.data.argsort()[0]
        uprank = map(int, rank[::-1])
        print uprank
        #print y.data[0]
        """
        
            
        train = AnazlizeTrainer(dim_in, dim_mid, dim_out)
        train.load(self.model_path + '/train_analize')
        #y = train.predict(msgidlist[::-1])
        
         
        targetlist = msgidlist
        resultlist = np.zeros(dim_out)  ## 確率結果の最大値を格納する配列
        print resultlist
        for i in range(len(targetlist)):
            target = targetlist[i:]
            y = train.predict(target[::-1])
            print target
            print y.data[0]
            for i in range(len(y.data[0])):
                if y.data[0][i] > resultlist[i]:
                    resultlist[i] = y.data[0][i]
                
        print resultlist
        #print y.data.argmax(1)[0]
        #rank = y.data.argsort()[0]
        rank = resultlist.argsort()
        uprank = map(int, rank[::-1])
        print uprank
        #print y.data[0]
        
        result = []
        for i in uprank:
            print '%d, %2f' % (i, resultlist[i])
            item = {
                'score': round(float(resultlist[i]) * 100, 2),
                'id': i,
                'label': errjson[i]['label']
            }
            result.append(item)

        
        return result
    
            
    

"""
テスト用
python ./model.py [train|predict]
"""
if __name__ == '__main__':
    import sys
    arg1 = sys.argv[1] if len(sys.argv) == 2 else None 

    ## 予測
    if arg1 == 'predict':
        filename = sys.argv[2] if len(sys.argv) >= 3 else 'train_test.csv'
        try:
            model = AnalogModel()
            #data = model.loadCSV(filename)[0]
            data = 'cmapeerdが停止したためプロセスを再起動します\nエージェントが可能な変更を行った後動作を開始しました\ncmapeerdの自動再起動に失敗しました'
            result = model.predict(data)
            print 'predict: ', data
            print 'result: ',  result
        
        except Exception as exce:
            print traceback.format_exc()
    
    
    else:
        print('usage: ' + __file__ + ' [train | predict]')