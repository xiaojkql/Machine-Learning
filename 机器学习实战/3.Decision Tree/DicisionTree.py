#!/usr/bin/python
# coding:utf-8

from __future__ import print_function

print(__doc__)
import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter


def createDataSet():
    '''
    @param
    @return
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    :param dataSet:
    :return:
    '''
    numEntries = len(dataSet)
    labelCounts = {} # 初始化使用
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, index, value):
    '''
    提取符合第index特征 = value的数据集列表
    :param dataSet:
    :param index:
    :param value:
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index] # 剔除axis的内容
            reducedFeatVec.extend(featVec[index + 1:]) # extend 与 append两个方法的差别
            retDataSet.append(reducedFeatVec)
    # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    通过计算信息增益找到最好的划分特征
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1 # 最后一列为标签
    baseEntropy = calcShannonEnt(dataSet) # 计算该数据集总的香浓熵
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 记录该特征下在该数据集中所具有的具体值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    当特征用完时，用此函数来判别所属类别，策略：少数服从多数
    :param classList:
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    # major_label = Counter(classList).most_common(1)[0]
    # return major_label


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 仅有一个类别时停止返回类别
        return classList[0]
    if len(dataSet[0]) == 1: # 特征用完了，停止，用多数服从少数的规则
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet) # 特征编号
    bestFeatLabel = labels[bestFeat] # 特征名称
    myTree = {bestFeatLabel: {}} # 节点
    del (labels[bestFeat]) # 因为该标签已经使用了，所以剔除
    featValues = [example[bestFeat] for example in dataSet] # 获取最优特征的数据集
    uniqueVals = set(featValues)  # 获取最优特征下的属性值
    for value in uniqueVals:
        subLabels = labels[:]
        # 递归 该特征值作为节点的值要么是一颗子树，要么就是类别标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''

    :param inputTree: 决策树
    :param featLabels: 决策树的标签
    :param testVec: 测试数据
    :return:
    '''
    # 获取tree的根节点对于的key值
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr] # 获得该节点标签对应的value
    featIndex = featLabels.index(firstStr) # 获取该特征在特征向量中的位置
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex] # 节点对应的特征对应的测试数据的值
    valueOfFeat = secondDict[key] # 根据上一个测试数据在该特征上的取值获取下一颗子树
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict): # 如果该特征对应的value仍然是一个dict则表明下面继续是一颗子树
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    # -------------- 第一种方法 start --------------
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
    # -------------- 第一种方法 end --------------

    # -------------- 第二种方法 start --------------
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    # -------------- 第二种方法 start --------------


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 获得树的高度
    print(get_tree_height(myTree))

    # 画图可视化展现
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    '''

    :return:
    '''
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lensData.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)


def get_tree_height(tree):
    '''
    :param tree:
    :return:
    '''
    if not isinstance(tree, dict):
        return 1
    child_trees = tree.values()[0].values()
    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)
        if child_tree_height > max_height:
            max_height = child_tree_height
    return max_height + 1


if __name__ == "__main__":
    fishTest()
    ContactLensesTest()