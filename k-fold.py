# coding=gbk
#import matplotlib.pyplot as plt
import struct
import random
import scipy
import h5py
import operator
from numpy import *
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, Activation, AveragePooling2D
from keras.optimizers import SGD
from sklearn.model_selection import KFold



#把地址和类型的标签转换成数字索引
def shopInfo2Labels(filename):
    fr = open(filename)
    Lines =  fr.readlines()
    cities = {'':0}         #城市标签
    places = {'':0}         #位置标签
    kind1 = {'':0}          #第1种类标签
    kind2 = {'':0}          #第2种类标签
    kind3 = {'':0}          #第3种类标签
    nl = len(Lines)
    m = len(Lines[0].split(','))
    Mat = range(0,nl)
    index = 0

    for line in Lines:
        #change labels into number index
        wlist = line.split(',')
        #print wlist[1], wlist[7], wlist[8]
        if wlist[1] not in cities:
            cities[wlist[1]] = len(cities)
        if wlist[2] not in places:
            places[wlist[2]] = len(places)
        if wlist[7] not in kind1:
            kind1[wlist[7]] = len(kind1)
        if wlist[8] not in kind2:
            kind2[wlist[8]] = len(kind2)
        if wlist[9] not in kind3:
            kind3[wlist[9]] = len(kind3)

        #wlist = wlist.strip()         #移除头尾的指定字符串（默认空格
        wlist[1] = cities[wlist[1]]
        wlist[2] = places[wlist[2]]
        wlist[7] = kind1[wlist[7]]
        wlist[8] = kind2[wlist[8]]
        wlist[9] = kind3[wlist[9]]
        for i in [0,3,4,5,6]:
            if wlist[i] == '':
                wlist[i] = 0
            else:
                wlist[i] = int(wlist[i])

        Mat[index] = wlist
        #clslabelVector .append(int(listFromline[-1]))           #输出最后一个元素标签哦
        index += 1
    print "Get the infos of all shops!"
    return Mat,cities, places,kind1,kind2,kind3,nl





#//////////////////////////////////////////////////////////////////////////////////////

def ExviewInfo2File(filename, CntMat, month):
    fr = open(filename)
    while True:
        line = fr.readline()
        if len(line) == 0:
            break
        words = line.split(',')
        time = words[2].split(' ')
        dateStr = time[0].split('-')
        dateNum = (int(dateStr[0])-2015)*365 + month[int(dateStr[1])] + (int(dateStr[2])) - 212   #从2015年7月开始
        CntMat[int(words[1])-1][dateNum-1] += 1
#        print words[1]
#        print viewCntMat[int(words[1])-1]

    print "Mat sorted and write!"
    return CntMat




#//////////////////////////////////////////////////////////////////////////////////////
#浏览
def view2Mat(filename):
    fr = open(filename)
    lines = fr.readlines()

    viewMat = []
    viewCntMat = []
    sum = 0
    print "Reading the views records...."
    for line in lines:
        shopTimeVect = []
        shopCntVect = []
        views = line.split(' ')
        cnt = 0
        prv_tm = -1
        sum += 1
        for view in views:

            view = view.split(',')
            if len(view) < 2:       #特殊处理错误格式
                continue
            view[0] = int(view[0])
            view[1] = int(view[1])
            shopTimeVect.append([view[0], view[1]])         #插入（日期，访问记录）
            if view[0] != prv_tm:
                if prv_tm != -1:
                    shopCntVect.append([prv_tm, cnt])          #插入(日期，访问次数)
                prv_tm= view[0]
                cnt = 1
            else:
                cnt += 1
        viewMat.append(array(shopTimeVect))
        viewCntMat.append(array(shopCntVect))
    return array(viewMat), array(viewCntMat)





#//////////////////////////////////////////////////////////////////////////////////////
#生成浏览和付款的矩阵信息（日期，量）
def cntviews(filename):
    fr = open(filename)
    cnt = 0
    payCntMat = zeros([2000,500],int32)
    while True:
        cnt += 1
        print cnt
        line = fr.readline()
        line = line[0:-2]
        if len(line) == 0:
            break
        times = line.split(' ')
        print times
        for tm in times:
            tm = tm.split(',')
            payCntMat[int(tm[0])-1][int(tm[1])] += 1

    fr.close()
    print "ViewCntMatrix got! max = ", max
    return  payCntMat

def cntpays(filename):
    fr = open(filename)
    cnt = 0
    max = int32(0)
    payCntMat = zeros([2000,500],int32)
    fr.readline()
    while True:
        cnt += 1
        print cnt
        line = fr.readline()
        line = line[0:-2]
        if len(line) == 0:
            break
        times = line.split(' ')
        for tm in times:
            tm = tm.split(',')
            payCntMat[int(tm[0])-1][int(tm[1])] += 1
            if payCntMat[int(tm[0])-1][int(tm[1])] > max:
                max = payCntMat[int(tm[0])-1][int(tm[1])]
    fr.close()
    print "PayCntMatrix got! max = ", max
    return  payCntMat

def writeCntMat(mat, filename):
    fw = open(filename, 'w')
    max = 0
    for i in range(2000):
        line = ''
        for j in range(500):
            line += str(i) + ',' + str(mat[i][j]) + ' '
            if mat[i][j] > max:
                max = mat[i][j]
        line = line[0:-1]
        line += '\n'
        fw.write(line)
    print "max = ",max
    fw.close()



def readCntMat(filename):
    maxx = 0
    fr = open(filename, 'r')
    Mat = zeros([2000,500],int32)
    for i in range(2000):
        line = fr.readline()
        day = line.split(' ')
        for j in range(499):
            val =day[j].split(',')
            Mat[i][j] = float(val[1])
            if(Mat[i][j] > maxx):
                maxx = Mat[i][j]
        val =day[499][0:-1]
        val = val.split(',')
        Mat[i][499] = float(val[1])
    fr.close()
    print "maxx = ", maxx
    return Mat

#//////////////////////////////////////////////////////////////////////////////////////
#选择不同的类型的店铺，输出其销量和浏览矩阵
def clsCntMat(cities, places, kind1, kind2, kind3, shoplist, score):#,viewCntMat,payCntMat):
    shops = shoplist
    shops_city = []
    shops_loca = []
    shops_pay = []
    shops_score = []
    shops_comm = []
    shops_level = []
    shops_kind1 = []
    shops_kind2 = []
    shops_kind3 = []
#//////////////////////////////////////////////////////////
#    strg = '[2]cities: '
    cnt = 0
    for pl in cities:
#        strg += str(cnt) +'.'+ pl + ' '
        cnt += 1
 #   sel = raw_input(strg)
    sel = '*'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops:
            if sh[1] == int(sel):
                shops_city.append(sh)
    else:
        shops_city = shops
    print len(shops_city)
#/////////////////////////////////////////////////////////
 #   strg = "[3]Location: "
    cnt = 0
    for pl in places:
#        strg += str(cnt) +'.'+pl + ' '
        cnt += 1
#    sel = raw_input(strg)
    sel = '*'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops_city:
            if sh[2] == int(sel):
                shops_loca.append(sh)
    else:
        shops_loca = shops_city
    print len(shops_loca)
#/////////////////////////////////////////////////////////
#    strg = "[4]Pay:  1 - 4 >> "
#    sel = raw_input(strg)
    sel = '*'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops_loca:
            if sh[3] == int(sel):
                shops_pay.append(sh)
    else:
        shops_pay = shops_loca

    print len(shops_pay)
#/////////////////////////////////////////////////////////
#    strg = "[5]score:  0 - 4 >> "
#    sel = raw_input(strg)
#    if sel == 'q':
#        return []
    sel = score

    if sel != '*':
        for sh in shops_pay:
            if sh[4] == int(sel):
                shops_score.append(sh)
    else:
        shops_score = shops_pay

    print len(shops_score)
#/////////////////////////////////////////////////////////
#    strg = "[6]comm:  0 - 4 >> "
#    sel = raw_input(strg)
    sel = '*'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops_score:
            if sh[5] == int(sel):
                shops_comm.append(sh)
    else:
        shops_comm = shops_score
    print len(shops_comm)
#/////////////////////////////////////////////////////////
#    strg = "[7]level:  0 - 4 >> "
#    sel = raw_input(strg)
    sel = '*'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops_comm:
            if sh[6] == int(sel):
                shops_level.append(sh)
    else:
        shops_level = shops_comm

    print len(shops_level)
    return shops_level



def trainModelRg1(train_data, train_label, test_data, test_label):

    model = Sequential()

    # feature 0: 80 days linear    
    c1 = Conv2D(filters=16, kernel_size=(2,3),  strides=(2,2), padding='SAME', input_shape = shape(train_data[0]))
    model.add(c1)
    model.add(Activation('relu'))
    #c2 = Conv2D(filters=20, kernel_size=(2,3), strides=(2,2))
    #model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))
    
    sgd = SGD (lr=0.00000002, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_data, train_label, batch_size = len(train_data) / 310, epochs =250, validation_data=(test_data, test_label))

    return model


def trainModelRg2(train_data, train_label, test_data, test_label):

    model = Sequential()

    # average feature 0: 80 days linear    
    avg = AveragePooling2D((1,3), (1,2), 'same', input_shape = shape(train_data[0]))
    model.add(avg)
    c1 = Conv2D(filters=21, kernel_size=(2,3),  strides=(2,2), padding='SAME', input_shape = shape(train_data[0]))
    model.add(c1)
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    sgd = SGD (lr=0.00000002, decay=3e-4, momentum=0.9, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_data, train_label, batch_size = 10, epochs =200, validation_data=(test_data, test_label))

    return model



def trainModelLin3(train_data, train_label, test_data, test_label):

    model = Sequential()

    model.add(Flatten(input_shape = shape(train_data[0])))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))

    sgd = SGD (lr=0.00000005, decay=8e-4, momentum=0.7, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_data, train_label, batch_size = 10, epochs =220, validation_data=(test_data, test_label))
    return model

#////////////////////////////////////////////////////////////////////////////////

shop_info = "meta/shop_info.txt"
user_view = "meta/user_view.txt"
user_pay = "meta/user_pay.txt"
extra_view = "meta/extra_user_view.txt"
month = [0,0,31,59,90,120,151,181,212,243,273,304,334]

#//test!
#mat = zeros([2000,500],int32)
#payCntMat = ExviewInfo2File(user_pay, mat, month)
#viewCntMat = ExviewInfo2File(user_view, mat, month)
#viewCntMat = ExviewInfo2File(extra_view, viewCntMat, month)

#step 1`
#viewInfo2File(user_view, shopNum)      #把shop_view的购买信息按商店排序输出到文件中
#payInfo2File(user_view)         #把shop_pay文件中的购买信息按商店排序输出到文件中，一行代表一个商店的信息
#viewMat, viewCntMat = view2Mat("shop_histry11.txt")        #从商店浏览信息获取记录，然后统计每天购买的次数
#mat = zeros([2000,500],int32)
#payCntMat = ExviewInfo2File(user_pay, mat)


#step 2
#viewCntMat = cntviews("meta/shop_views.txt")
#payCntMat = cntpays("meta/shop_pays.txt")
#writeCntMat(viewCntMat, "meta/viewsMat.txt")
#writeCntMat(payCntMat, "meta/paysMat.txt")


#step 3
viewCntMat = readCntMat("meta/viewsMat.txt")
payCntMat = readCntMat("meta/paysMat.txt")
#viewCntMat = ExviewInfo2File(extra_view, viewCntMat)

print "get the CntMat!"

cnt_zeros = 0
for ii in range(2000):
    for jj in range(1, 443):
        if payCntMat[ii][jj-1] >= 6 and payCntMat[ii][jj+1] >= 6 and payCntMat[ii][jj]==0:
            payCntMat[ii][jj] = (payCntMat[ii][jj-1] + payCntMat[ii][jj+1])/2
            cnt_zeros += 1

print "pay fill 0:", cnt_zeros

cnt_zeros = 0
for ii in range(2000):
    for jj in range(1, 443):
        if viewCntMat[ii][jj-1] >= 3 and viewCntMat[ii][jj+1] >= 3 and viewCntMat[ii][jj]==0:
            viewCntMat[ii][jj] = (viewCntMat[ii][jj-1] + viewCntMat[ii][jj+1])/2
            cnt_zeros += 1

print "view fill 0:", cnt_zeros
print payCntMat[800]
print payCntMat[500]
print payCntMat[700]
print payCntMat[600]

#step 4
dataMat, cities, places, kind1, kind2, kind3, shopNum = shopInfo2Labels(shop_info)
print len(dataMat)
#for ll in dataMat:
#    print ll[4]


#显示绘图0
'''
num = 0
while True:
    shops_sel = clsCntMat(cities,places, kind1, kind2, kind3,dataMat)
    while  True:
        num = raw_input("shopNum? ")
        if num == 'n':
            break
    #    fig = plt.figure()
        plt.plot(range(500), viewCntMat[int(num)-1])
        plt.show()
        plt.plot(range(500), viewCntMat[int(num)-2])
    #    ax = fig.add_subplot(111)
    #    ax.scatter(viewCntMat[int(num)][:,0], viewCntMat[int(num)][:,1], 20, 15*viewCntMat[int(num)][:,1])
        plt.show()
        plt.close()
'''

#分类显示折线图
'''
cnt = 0
while True:
     shops_sel = clsCntMat(cities, places, kind1, kind2, kind3,dataMat)
     num = len(shops_sel)

     for i in range(0, num):
         plt.figure(2*i)
         plt.plot(range(500), payCntMat[shops_sel[i][0]])
         plt.figure(2*i+1)
         plt.plot(range(500), viewCntMat[shops_sel[i][0]])

     plt.show()
     plt.close()
'''


score = 0
modelFeatureFile = open("model/modelFeat.txt", 'w')
kfoldFile = open("model/kfold.txt", 'w')
while score <= 4:
    print "-------------------------------Now preducing train-model of score = ", score
    # train
    percase = 14        #14 days of one shop
    day = 2
    start = 371 +day- percase   # start from 0
    end = 448 +day- percase
    
#    # final attempt
#    percase = 1
#    start = 372 - percase   # start from 0
#    end = 449 - percase

    cnt = 0

    shops_sel = clsCntMat(cities, places, kind1, kind2, kind3, dataMat, score)  #get the selected shops list
    if len(shops_sel)==0:
        break
    #write the shops number
    num = len(shops_sel)
    train_num = (num*3/4+1)
    test_num = (num - train_num)



#----------------training data sets---------------------
    #write dataset number
    print "train=",train_num * percase
    print "test=",test_num * percase

    shff_Mat = [[0,0,0,0,0,0,0,0] for i in range(num*percase)]


#//////////store shop-offsetday matrix///////////////
    for i in range(0, num):
        for j in range(0, percase):
            shff_Mat[i*percase + j][0] = shops_sel[i][0]-1
            shff_Mat[i*percase + j][1] = j
            shff_Mat[i*percase + j][2] = shops_sel[i][3]        #per_pay
            shff_Mat[i*percase + j][3] = shops_sel[i][4]        #score
            shff_Mat[i*percase + j][4] = shops_sel[i][5]        #comment
            shff_Mat[i*percase + j][5] = shops_sel[i][6]        #level
            shff_Mat[i*percase + j][6] = shops_sel[i][7]        #kind 1
            shff_Mat[i*percase + j][7] = shops_sel[i][8]        #kind 2


    random.shuffle(shff_Mat)
#//////////////shuffle the train and test dataset Matrix//////////////////////
#    random.shuffle(shff_Mat)
#    trainMat = shff_Mat[0 : train_num*percase]
#    testMat = shff_Mat[train_num*percase : num*percase]

#///////////////////////////////train sets//////////////////////////////////

    all_data = []
    all_label = []
    all_shp = []
    all_sum = []
    all_cyc = []
    all_div = []
    all_j = []
    for shp in shff_Mat:
        j = shp[1]
        all_j.append(j)

        #model 2's feature
        matrx = [[], []]
        for k in range(1, 13):
            matrx[0].append(viewCntMat[shp[0]][end+j-k*7]);
            matrx[1].append(payCntMat[shp[0]][end+j-k*7]);

        all_cyc.append([matrx])


        #model 0/1 's feature
        matrx = [0, 0]
        atrb = array([shp[2], shp[3], shp[4]])  #per_pay, score, comment
        matrx[0] = append(viewCntMat[shp[0]][start+j:end+j], atrb)
        atrb = array([shp[5], shp[6], shp[7]])  #level, kind1, kind2
        matrx[1] = append(payCntMat[shp[0]][start+j:end+j], atrb)

        lb = payCntMat[shp[0]][end+j]

        all_shp.append(shp[0])
        all_data.append([matrx])
        all_label.append(lb)




    all_data = array(all_data)
    all_cyc = array(all_cyc)
    all_label = array(all_label)
    all_shp = array(all_shp)
    all_j = array(all_j)

    #split the data into K-folds
    kf = KFold(n_splits=4)
    indx = 0
    for train, test in kf.split(all_data):
        res = [[], [], []]
        train_data = all_data[train]
        train_label = all_label[train]
        train_cyc = all_cyc[train]
        test_data = all_data[test]
        test_label = all_label[test]
        test_cyc = all_cyc[test]
        print "K-fold's", indx, " iteration\n"

#///////////////////////////////////////////////////////////////////////////////////////////
        model1 = trainModelRg1(train_data, train_label, test_data, test_label)
        model1.save("model/md-score"+str(score)+'.'+str(indx))
        prd = model1.predict(test_data,  verbose=1)
        err = 0
        for i in range(len(prd)):
            val = prd[i]
            res[0].append(val)
            err += (abs(val-test_label[i])/(val+test_label[i]))
        print "-----------------------------------model 1's err = ", err/len(prd)



#///////////////////////////////////////////////////////////////////////////////////////////
        model2 = trainModelRg2(train_data, train_label, test_data, test_label)
        model2.save("model/md-avg"+str(score)+'.'+str(indx))
        prd = model2.predict(test_data,  verbose=1)
        err = 0
        for i in range(len(prd)):
            val = prd[i]
            res[1].append(val)
            err += (abs(val-test_label[i])/(val+test_label[i]))
        print "-----------------------------------model 1's err = ", err/len(prd)



#///////////////////////////////////////////////////////////////////////////////////////////
        model3 = trainModelLin3(train_cyc, train_label, test_cyc, test_label)
        model3.save("model/md-7days"+str(score)+'.'+str(indx))
        prd = model3.predict(test_cyc,  verbose=1)
        for i in range(len(prd)):
            val = prd[i]
            res[2].append(val)
            err += (abs(val-test_label[i])/(val+test_label[i]))
        print "-----------------------------------model 1's err = ", err/len(prd)

        indx += 1


        # output the array
        for i in range(len(res[0])):
            line = str(all_shp[test][i])
            line += ' ' + str((all_j[test])[i])
            line += ' ' + str(all_label[test][i])
            line += ' ' + str(res[0][i])
            line += ' ' + str(res[1][i])
            line += ' ' + str(res[2][i]) + '\n'
            modelFeatureFile.write(line)
#///////////////////////////////////////////////////////////////////////////////////////////
        
        for val in train:
            kfoldFile.write(str(val)+' ')
        kfoldFile.write('\n')

        for val in test:
            kfoldFile.write(str(val)+' ')
        kfoldFile.write('\n')

    score += 1


modelFeatureFile.close()
kfoldFile.close()



#import scipy
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Input, Dense, Dropout, Flatten
#from keras.layers import Conv2D, Activation, AveragePooling2D
#from keras.optimizers import SGD
#import keras.layers.pooling.AveragePooling2D

