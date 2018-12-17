# -*- coding:utf-8 -*-
import numpy as np
import math

def loadDataSet():
    '''
    创建实验文档，及其类别
    :return:
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    '''
    从实验文档中创建词汇列表
    :param dataSet:
    :return:
    '''
    vocabList = set([])
    for data in dataSet:
        vocabList = vocabList | set(data)
    return list(vocabList)

def setOfWords2Vec(vocabList,inputSet):
    '''
    给输入文档根据词向量列表创建属于该文档的词向量
    :param vocabList:
    :param inputSet:
    :return:
    '''
    vecOfInput = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vecOfInput[vocabList.index(word)] = 1
    return vecOfInput

def bagOfWord2Vec(vocabList,inputSet):
    vecOfInput = [0] * len(vocabList)
    for data in inputSet:
        vecOfInput[vocabList.index(data)] += 1
    return vecOfInput

# 上面的函数的目的是将一串文字的文档转换为数字的向量，然后根据此计算条件概率

# 计算先验概率知识
def trainNB0(trainMat,trainCat):
    '''
    计算每个类别概率 一个概率值
    计算每个类别下，某一特征的某一特征值出现的概率,一个概率向量，
    计算方法是计算该类别下，该词出现的次数与所有词出现的总次数之间的比值
    :param trainMat:
    :param trainCat:
    :return:
    '''
    # 计算正例类别概率
    pPositive = sum(trainCat)/float(len(trainCat))
    # 计算条件概率
    numPosi = np.ones(len(trainMat[0])) # 初始化为1 而不是初始化为0
    numNega = np.ones(len(trainMat[0])) # 保证每个单词都出现了，不会有条件概率为零情况产生
    sumPosi = 2; sumNega = 2
    for i in range(len(trainCat)):
        if trainCat[i] == 1:
            numPosi += trainMat[i]
            sumPosi += sum(trainMat[i])
        else:
            numNega += trainMat[i]
            sumNega += sum(trainMat[i])
    pPosi = np.log(numPosi/float(sumPosi)) # 使用对数，避免相乘时数据变得 ln(ab) = lna + lnb
    pNega = np.log(numNega/float(sumNega))
    return pPositive,pPosi,pNega

def classifyNB(docVec,pPosi,pNega,pPositive):
    '''
    p(c|w) = p(w|c)p(c) / p(w) = p(w_1/c)*p(w_2/c)...p(w_n|c)*p(c) / p(w) 取对数后就是相加
    :param docVec:
    :param pPosi:
    :param pNega:
    :param pPositive:
    :return:
    '''
    pClass1 = sum(docVec * pPosi) + np.log(pPositive) # 这里计算的是某个词向量中某个词出现的概率，
    pClass0 = sum(docVec * pNega) + np.log(1.0-pPositive)
    if pClass1>pClass0:
        return 1
    else:
        return 0

def testNB():
    '''

    :return:
    '''
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for data in listOPosts:
        vec = setOfWords2Vec(myVocabList,data)
        trainMat.append(vec)
    pP,p1V,p0V = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = setOfWords2Vec(myVocabList,testEntry)
    print(classifyNB(np.array(thisDoc),p0V,p1V,pP))

    testEntry = ['stupid','garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(classifyNB(np.array(thisDoc), p1V,p0V, pP))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '''
    Desc:
        对贝叶斯垃圾邮件分类器进行自动化处理。
    Args:
        none
    Returns:
        对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    # 随机取 10 个邮件用来测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pSpam,p1V, p0V,  = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p1V, p0V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))

if __name__ == '__main__':
    testNB()
    spamTest()


