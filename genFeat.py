# coding=gbk
#import matplotlib.pyplot as plt
import struct
import random
from numpy import *
import operator

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
#把每个商店的浏览信息输出到矩阵文本当中

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
#把付款信息从原文件读取，按日期排序后输出到文件中
def payInfo2File(filename):
    fr = open(filename)
    shopViews = []                #初始化所有商店的访问记录数组
    line = 'start'
    lstshp = 2222
    while True :
        line = fr.readline()
        if len(line) == 0:
            break
        words = line.split(',')
        time = words[2].split(' ')
        dateStr = time[0].split('-')
        dateNum = (int(dateStr[0])-2015)*365 + (int(dateStr[1]))*30 + (int(dateStr[2])) - 210   #从2015年7月开始
        shp_num = int(words[1])
        if shp_num != lstshp and lstshp <= 2000:
            fw = open("meta/shop_views1.txt", 'a')
            shopViews.sort()
            wline = ''
            for sh in shopViews:
                wline += sh[1] + ',' + str(sh[0]) + ' '
            wline += '\n'
            fw.write(wline)
            fw.close()
            shopViews = []
        lstshp = shp_num
        shopViews.append([dateNum,words[1]])
    print "Mat sorted and write!"
    fw.close()

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
            Mat[i][j] = int(val[1])
            if(Mat[i][j] > maxx):
                maxx = Mat[i][j]
        val =day[499][0:-1]
        val = val.split(',')
        Mat[i][499] = int(val[1])
    fr.close()
    print "maxx = ", maxx
    return Mat

#//////////////////////////////////////////////////////////////////////////////////////
#选择不同的类型的店铺，输出其销量和浏览矩阵
def clsCntMat(cities, places, kind1, kind2, kind3, shoplist):#,viewCntMat,payCntMat):
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
    strg = '[2]cities: '
    cnt = 0
    for pl in cities:
        strg += str(cnt) +'.'+pl + ' '
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
    strg = "[3]Location: "
    cnt = 0
    for pl in places:
        strg += str(cnt) +'.'+pl + ' '
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
    strg = "[4]Pay:  1 - 4 >> "
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
    strg = "[5]score:  0 - 4 >> "
#    sel = raw_input(strg)
    sel = '0'
    if sel == 'q':
        return []

    if sel != '*':
        for sh in shops_pay:
            if sh[4] == int(sel):
                shops_score.append(sh)
    else:
        shops_score = shops_pay

    print len(shops_score)
#/////////////////////////////////////////////////////////
    strg = "[6]comm:  0 - 4 >> "
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
    strg = "[7]level:  0 - 4 >> "
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


#////////////////////////////////////////////////////////////////////////////////

shop_info = "meta/shop_info.txt"
user_view = "meta/user_view.txt"
user_pay = "meta/user_pay.txt"
extra_view = "meta/extra_user_view.txt"
month = [0,0,31,59,90,120,151,181,212,243,273,304,334]

#//test!


#step 3
viewCntMat = readCntMat("meta/viewsMat.txt")
payCntMat = readCntMat("meta/paysMat.txt")
#viewCntMat = ExviewInfo2File(extra_view, viewCntMat)


#output the label
#flb = open("feature/label", 'w')
#for v in payCntMat:
#    line = ""
#    for pay in v[448: 448+14]:
#        line += ' ' + str(pay)
#    flb.write(line+'\n')
#flb.close()


flb = open("feature/3months", 'w')
for oneshp in payCntMat:
    for j in range(7):      #use a week data
        sum = [0,0,0]
        for k in range(90):     #use the data of last 90 days
            sum[k/30] += oneshp[448-1-k+j]
        sum[0] /= 30
        sum[1] /= 30
        sum[2] /= 30
        flb.write(' '+str(sum[2])+','+str(sum[1])+','+str(sum[0]))
    flb.write('\n')
flb.close()


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




while True:

    # train
#    percase = 14        #14 days of one shop
#    start = 371 - percase   # start from 0
#    end = 448 - percase
    
#    # final attempt
    percase = 1
    start = 372 - percase   # start from 0
    end = 449 - percase

    cnt = 0

    shops_sel = clsCntMat(cities, places, kind1, kind2, kind3, dataMat)  #get the selected shops list
    if len(shops_sel)==0:
        break
    #write the shops number
    num = len(shops_sel)
    train_num = (num*3/4+1)
    test_num = (num - train_num)



#----------------training data sets---------------------
    #write dataset number
    print "train=", train_num * percase
    print "test=", test_num * percase

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


#//////////////shuffle the train and test dataset Matrix//////////////////////
    random.shuffle(shff_Mat)
    trainMat = shff_Mat[0 : train_num*percase]
    testMat = shff_Mat[train_num*percase : num*percase]

#///////////////////////////////train sets//////////////////////////////////

    all_data = []
    all_label = []
    all_shp = []


    for shp in shff_Mat:
        j = shp[1]

        matrx = [[], []]
        for k in range(1, 13):
            matrx[0].append(viewCntMat[shp[0]][end+j-k*7]);
            matrx[1].append(payCntMat[shp[0]][end+j-k*7]);


        cnt += 1
        lb = payCntMat[shp[0]][end+j]

  #      print "label:",lb

        all_shp.append(shp[0])
        all_data.append([matrx])
        all_label.append(lb)

    all_data = array(all_data)
    all_label = array(all_label)

    test_data = all_data[train_num*percase : num*percase]
    test_label = all_label[train_num*percase : num*percase]
    train_data = all_data[0 : train_num*percase]
    train_label = all_label[0 : train_num*percase]


#    print test_data, test_label

#    ex = raw_input("Finish select and output!\n")
    ex = 'q'
    if ex == 'q':
        break




#import scipy
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, Activation, AveragePooling2D
from keras.optimizers import SGD
from keras.models import load_model
import h5py
#import keras.layers.pooling.AveragePooling2D

print "shape:", shape(train_data), shape(train_label)
print type(train_data), type(train_label)

ex = raw_input("train / predict ? ")
if ex == 't':

    model = Sequential()

    ''' feature 0: 80 days linear    
    c1 = Conv2D(filters=16, kernel_size=(2,3),  strides=(2,2), padding='SAME', input_shape = shape(train_data[0]))
    model.add(c1)
    model.add(Activation('relu'))
    #c2 = Conv2D(filters=20, kernel_size=(2,3), strides=(2,2))
    #model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))
    
    #model.add(Flatten(input_shape = shape(train_data[0])))
    #model.add(Dense(1))
    
    sgd = SGD (lr=0.00000002, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_data, train_label, batch_size = 8, epochs =200, validation_data=(test_data, test_label))
    model.save("model/md0")
    '''

    # average feature 0: 80 days linear    
    model.add(Flatten(input_shape = shape(train_data[0])))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))
    
    #model.add(Flatten(input_shape = shape(train_data[0])))
    #model.add(Dense(1))
    
    sgd = SGD (lr=0.00000002, decay=1e-4, momentum=0.7, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_data, train_label, batch_size = 8, epochs =200, validation_data=(test_data, test_label))
    model.save("model/md3")

else:

#    model= load_model("model/md-score3")
    model= load_model("model/md3")
    prd = model.predict(all_data,  verbose=1)
    res = []
    for val in prd:
        res.append(val[0])
    err = sum(abs(res-all_label)/(res+all_label)) / len(all_label)
    print err
   
    featfile = open("feature/prv", 'a');
    for nn in range(len(res)):
        featfile.write(str(all_shp[nn]) + ' ' + str(res[nn])+'\n')
