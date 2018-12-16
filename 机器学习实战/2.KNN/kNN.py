# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import operator
import collections
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    '''
    :param inX: 要分类的点的特征向量
    :param dataSet: 已知数据集
    :param labels: 已知数据集的类别
    :param k: 取几近邻
    :return:
    '''
    dataSetSize = dataSet.shape[0] # .shape是一个list 可以用slice进行取值

    # a = [1,2] b = np.tile(a,(2,2)) b = [[1,2,1,2],[1,2,1,2]]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2 # element wise power
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() # 提取 np.array 中数据从小到大排列后的索引
    classCount = {} # 创建一个空字典 使用之前一定要先创建
    for i in range(k): # range(start,stop,step) 选择最小的k的点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 此处需要使用一个dict 所以前面提前定义了一个空的
        # dict.get(key,default = value) 查找字典里面key，若存在返回对应的value，若不存在返回 default
    sortedClassCount = sorted(classCount.iteritems(),
                              key = operator.itemgetter(1),reverse = True) # 获取得分你最高的
    # iteritems生成一个列表迭代器 dict = {1:2,2:3} dict.item() : [(1,2),(2,3)]
    # operator提供的函数itemgetter()函数的用法
    return sortedClassCount[0][0]

def classify_simple(inX,dataSet,labels,k):
    # 使用numpy广播
    diffMat = np.sum((inX - dataSet)**2,axis=1)**0.5
    # 使用列表解析
    k_labels = [labels[index] for index in diffMat.argsort()[0:k]]
    # 使用collections的counter及它的most_common
    # Counter返回的是一个字典，most_common返回的是一个列表，列表中的元为元组
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

def test1():
    group,labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
    print(classify_simple([0,0],group,labels,3))

def file2matrix(filename): # 数据读入的方法
    fr = open(filename)
    arrayOlines = fr.readlines() #一行一行的入，将一行当作一个整体存入一个列表之中
    print(arrayOlines)
    numberOfLines = len(arrayOlines) # 行数
    print(numberOfLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = [] # 同样要先定义个list对象，后面才能使用其方法append
    index = 0
    for line in arrayOlines:
        line = line.strip() # str.strip(char) 去除str首尾为char的字符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:-1]
        # 不能对这样一个不存在下标的list使用[][]添加元素，而要使用append方法
        # classLabelVector[index] = listFromLine[-1]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 作图的方式对数据进行分析
def analysis(returnMat,classLabelVector):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:,0],returnMat[:,1],
               15.0*np.array(classLabelVector),15.0*np.array(classLabelVector))
    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    # 或者直接使用element wise 减法与除法
    norm_DataSet = (dataSet - minVals) / ranges
    return normDataSet,minVals,ranges


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix("datingTestSet.txt")
    normMat,minVals,ranges = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: ",(errorCount/numTestVecs)) # 注意该除法，整数相除时，结果为0

def test2():
    datingClassTest()

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,minVals,ranges = autoNorm(datingMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person: ',resultList[classifierResult-1])

# 手写识别系统
# 读入图片将图片矩阵转换为向量
def imageVector(filename):
    returnVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline() # 直接将读入的行转换为一维list
        for j in range(32):
            returnVector[0,j+i*32] = int(line[j])
    return returnVector

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = imageVector('trainingDigits/%s'%fileNameStr)

    testFileList = listdir('testDigits')
    erroCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        numStr = int(fileStr.split('_')[0])
        vectorUnderTest = imageVector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the predict is: %d and the real is: %d'%(classifierResult,numStr))
        if(classifierResult != numStr):
            erroCount += 1.0
    print(erroCount)
    print(erroCount/mTest)

if __name__ == '__main__':
    test1()
    test2()
    # classifyPerson()
    handWritingClassTest()

