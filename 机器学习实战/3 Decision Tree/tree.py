# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import operator
from collections import Counter
import treePlotter

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers'] # 特征的标签名称
    return dataSet,labels

def calcShannonEnt(dataSet):
    '''
    param dataSet: 某个数据集
    return: 返回数据集的信息熵
    '''
    # 第一种紧凑的方法
    numOfEachCategory = Counter([data[-1] for data in dataSet])
    probs = [float(p[1])/len(dataSet) for p in numOfEachCategory.items()]
    shannonEnt_simple = sum([-p*math.log(p,2) for p in probs])

    # 第二种比较细致的
    numOfDataSet = len(dataSet)
    numEntries = {}
    for featVec in dataSet:
        numEntries[featVec[-1]] = numEntries.get(featVec[-1],0) + 1
    shannonEnt = 0.0
    for key in numEntries: # 遍历dict的方法
        probs_2= float(numEntries[key])/numOfDataSet
        shannonEnt += -probs_2 * math.log(probs_2,2)

    shannonEnt_1 = 0.0
    for key,value in numEntries.items(): # 遍历dict的方法
        probs_1= float(value)/numOfDataSet
        shannonEnt_1 += -probs_1 * math.log(probs_1,2)

    return shannonEnt_1

def splitDataSet(dataSet,axis,value):
    '''
    根据某个轴的某个值提取数据集
    param dataSet: 原始数据集
    param axis: 划分的轴
    param value: 轴的数据值
    return: retData 提取的数据集
    '''

    retData =[] # 注意这里添加数据集retData的方法append在list的最后添加元素
    for featVec in dataSet:
        if featVec[axis] == value:
            ret = featVec[0:axis]
            ret.extend(featVec[axis+1:])
            retData.append(ret)

    # 第二种紧凑方法
    # enumerate(sequence,start) 返回一个元组list(number,data_value)
    # 列表解析的方法应用 选择满足条件的
    retDataSetTemp = [data for data in dataSet for i,v in enumerate(data) if i == axis and v == value]
    retDataSet =[]
    for featVec in retDataSetTemp:
        ret = featVec[0:axis]
        ret.extend(featVec[axis+1:])
        retDataSet.append(ret)
    return retData


def chooseBestFeatureToSplit(dataSet):
    numOfFeatures = len(dataSet[0])-1
    baseEntro = calcShannonEnt(dataSet)
    bestGain = 0.0;bestFeature = 0
    for i in range(numOfFeatures):
        # 计算每个特征所具有的特征值
        featList = [data[i] for data in dataSet]
        uniqueFeat = set(featList) # 使用了集合，使元素独一无二
        newEntropy = 0.0
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = float(len(subDataSet))/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        if (baseEntro - newEntropy)>bestGain:
            bestGain = baseEntro - newEntropy
            bestFeature = i
    return bestFeature

# 使用比较紧凑的方法
def chooseBestFeature(dataSet):
    baseEntro = calcShannonEnt(dataSet)
    bestGain = 0.0
    for i in range(len(dataSet[0])-1):
        featCount = Counter([data[i] for data in dataSet]) # 得到一个counter发挥的巨大作用
        newEntropy = sum([float(feat[1])/len(dataSet)*calcShannonEnt(splitDataSet(dataSet,i,feat[0]))
                          for feat in featCount.items()])
        gain = baseEntro - newEntropy
        if(gain>bestGain):
            bestGain = gain
            bestFeatur = i
    return bestFeatur

def majority(classList):
    classCount = {}
    for line in classList:
        classCount[line[-1]] = classCount.get(line[-1],0) + 1
    sortedClass = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClass[0][0]

def majority_simple(classList):
    classCount = Counter([data[-1] for data in classList])
    # print(classCount) # classCount = {'no':3,'yes':2}
    # sortedClass = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    mostLabel = classCount.most_common(1)[0][0] # classCount.most_common(1) = [(key,value)]
    return mostLabel

def creatTree(dataSet,labels):
    # 第一种退出机制，所有元素的类别相同
    classList = [data[-1] for data in dataSet]
    # 使用列表的方法list.count(element)统计元素element在列表中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 第二种退出机制，所有特征都已经划分完整
    if len(dataSet[0]) == 1:
        return majority(dataSet)

    '''
    正常情况下创建
    首先选择最好的划分特征
    {bestFeatLabel:{value_1:{bestFeatLabel:{value_1:kind,value_2:{bestFeatLabel:{value_1:kind,value_2:kind}}}},value_2:kind}}
    '''
    bestFeat = chooseBestFeatureToSplit(dataSet)
    label = labels[bestFeat]
    myTree = {label:{}} # 下面要使用dict的切片形式，所以这里首先要创建一个空的
    featValues = [data[bestFeat] for data in dataSet]
    featValuesSet = set(featValues)
    for featValue in featValuesSet:
        subLabels = labels[:]
        del(subLabels[bestFeat])
        myTree[label][featValue] = creatTree(splitDataSet(dataSet,bestFeat,featValue),\
                                                subLabels)
    return myTree



def classify(inTree,featLables,testVec):
    '''
    应用已经建立好的决策树进行分类
    param inTree: 已构建的树
    param featLables: 特征标签
    param testVec: 测试数据向量
    return: 测试向量所属的类别
    '''
    print(featLables)
    firstLable = inTree.keys()[0]
    secondDict = inTree[firstLable]
    featIndex = featLables.index(firstLable) # 获得该特征标签在特征向量中的位置
    for key in secondDict.keys():
        if (testVec[featIndex] == key):
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLables,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def test1():
    dataSet, labels = createDataSet()
    myTree = creatTree(dataSet,labels)
    print(myTree)
    treePlotter.createPlot(myTree)
    testVec = [1,0]
    print(classify(myTree,labels,testVec))


# 使用pickle模块进行存储对象
def storeTree(inTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'r')
    return pickle.load(fr)

def lenTest():
    fr = open('lenses.txt')
    lenses = [line.strip().split('\t') for line in fr.readlines()]
    lenseTraining = lenses[0:int(len(lenses)*0.8)]
    lenLabel = ['age','prescript','astigmatic','tearRate']
    lenseTree = creatTree(lenseTraining,lenLabel)
    print(lenseTree)
    treePlotter.createPlot(lenseTree)
    storeTree(lenseTree,'Tree.txt')


if __name__ == '__main__':
    # test1()
    lenTest()