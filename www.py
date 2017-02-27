#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-
"""
Desc:   AnaLog
Author: Makoto OKITA
Date:   2016/10/16
Usage: ./analog.py ... flask WWWサーバ起動
"""
from flask import Flask, url_for
from flask import request, json, jsonify
import ConfigParser
import importlib
import os, sys, traceback, logging


## ログ
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
log.setLevel(logging.ERROR)
log.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

## import pathの追加
#print(os.getcwd())
#sys.path.append(os.getcwd())
sys.dont_write_bytecode = False  ## True/False pycを作らない
#os.chdir(os.path.dirname(__file__)) ## 実行ファイルのディレクトリを実行パスにする

## 設定値読み込み
## .pyと同じディレクトリにある.iniが設定ファイル
config = ConfigParser.SafeConfigParser()
config.read('www.ini')
http_host = config.get('FLASK','http_host')
http_port = config.getint('FLASK','http_port')
app_debug = config.getboolean('FLASK','app_debug')
log_level = getattr(logging, config.get('APP','log_level'))    ## LOG LEVEL!!
log.setLevel(log_level)


## Flask ##
app = Flask(__name__)


"""
### Error ###
400 Bad Request
403 Forbidden
404 Not Found
500 Internal Server Error
503 Service Unavailable
"""
## サーバエラーのトラップ
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(500)
def error_handler(error):
    #msg = 'Error: {code}\n'.format(code=error.code)
    #log.error(msg)
    #return msg, error.code
    return error

## アプリケーションエラー処理
def res_error(errno, msg, request):
    respmsg = {
        'status': False,
        'result': {
            'errnode': errno,
            'message': msg
        },
        'request': request
    }
    res = jsonify(respmsg)
    res.status_code = errno
    log.debug('response code:%d, data:%s', errno, respmsg)
    return res


"""
200 OK
"""
def res_ok(result):
    respmsg = {
        'status': True,
        'result': result
    }
    res = jsonify(respmsg)
    res.status_code = 200
    log.debug('response code:%d, data:%s', 200, respmsg)
    return res


"""
URL Routing
"""
## DNNの実行URL
@app.route('/analog/predict/<path:model>', methods = ['POST'])
def analog(model):
    log.debug('model: %s', model)
    log.debug('request data: %s', request.data)
    flag_slidematch = request.args.get('slidematch')
    print flag_slidematch   ## スライドしながらエラーケースをマッチさせるか
    req = request.data
    res = ''
    
    """
    res = 'hoge'
    return res_ok(res)
    """
    
    ## Modelのローディング
    try:
        modellib = 'model.' + model + '.model'## ロードするライブラリ
        log.debug('loading model: %s', modellib)
        module = importlib.import_module(modellib)
    except Exception as err:
        log.error('Model not found: %s', model, exc_info=True)
        return res_error(404, 'Model not found: ' + model, request.json)
    
    ## Modelの実行
    try:
        mymodel = module.AnalogModel()
        res = mymodel.predict(req)  ## 返り値にresultのJSON
    except Exception as err:
        log.error('Model execute error: ', exc_info=True)
        return res_error(500, 'Model execute error: ' + err.message, request.json)

    ## 正常終了
    log.info('rquest: %s', req)
    log.info('result: %s', res)
    
    return res_ok(res)
    

"""
Main
Python2.7だと、シングルスレッドのため、後から来た処理は待たされる
マルチスレッドオプションはPython3.xで利用可能
app.run(threaded=true)
または、Nginx/uWSGI等を使うことが好ましい
"""
if __name__ == '__main__':
    log.info('Starting AnaLog server...')
    log.debug("http_host=%s, http_port=%d", http_host, http_port)
    app.debug = app_debug   ## Debug Flag
    app.run(host=http_host, port=http_port) ## HTTP Server起動
