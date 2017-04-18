# coding=gbk

import xgboost as xgb
import random
from genFeatFromModel import *


# read in data
#dtrain = xgb.DMatrix('xgboost/demo/data/agaricus.txt.train')
#dtest = xgb.DMatrix('xgboost/demo/data/agaricus.txt.test')
dtrain = xgb.DMatrix('feature/train2/feature.txt')
dtest = xgb.DMatrix('feature/attempt/test.txt')


# specify parameters via map
param = {'max_depth':8, 'eta':0.3, 'silent':0, 'objective':'reg:linear' }
num_round = 30
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)


lb = dtest.get_label()
err = 0
for i in range(len(lb)):
    err += abs(preds[i]-lb[i])/(preds[i]+lb[i])
    print preds[i], lb[i]
print "err=", err/len(lb), len(lb)

##########################################################################
#modify the payCnt with
for i in range(len(preds)):
    payCntMat[shp[i]][end] = preds[i]



writeCntMat(payCntMat, "meta/paysMat.txt")
print "Update the payCntMat!\n"
