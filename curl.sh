#!/bin/bash
curl -v -X POST -H "Content-Type: text/plain" http://192.127.1.29:5001/analog/predict/default -d @curl.txt

## SOM
#curl -v -H "Content-type: application/json" -X POST http://localhost:5001/som/predict/default -d '{"param": [100,0,10,3,1,10,100,4,8,5,2]}'

## DNN
#curl -v -H "Content-type: application/json" -X POST http://localhost:5001/dnn/predict/default -d '{"param": [100,0,10,3,1,10,100,4,8,5,2]}'
