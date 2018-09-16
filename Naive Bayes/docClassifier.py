from numpy import *

# 准备数据：词表到向量的转换函数
# 创建实验样本
def loadDataSet():
    # 创建一些实验样本，即对词条进行切分后的文档集合
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字, 0 代表正常言论，这是一个类别标签集合
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建一个词汇表
def createVocabList(dataSet):
    # 创建一个空集，由于采用了set数据结构，故不会包含重复元素
    vocabSet = set([])
    # 遍历文档集合中的每一篇文档
    for document in dataSet:
        '''
        将文档列表转为集合的形式，保证每个词条的唯一性
        然后与vocabSet取并集，向vocabSet中添加没有出现
        的新的词条        
        '''
        vocabSet = vocabSet | set(document)
    # 将set转化为list以后返回
    return list(vocabSet)

# 输入词汇表以及文档，返回文档向量，向量的每个元素是1或者0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含各维度元素都是0的向量，和词汇表等长
    returnVec = [0]*len(vocabList)
    # 遍历文档，若出现了词汇表中的单词，则将输出的文档向量中对应的值设为1
    for word in inputSet:
        if word in vocabList:
            '''
            通过列表获取当前word的索引(下标)将词条向量中的对应下标的项由0改为1

            '''
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 验证转换函数功能
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
# print(myVocabList)

# 训练算法： 从词向量计算概率
# 朴素贝叶斯分类器训练函数，trainMatrix为文档矩阵，trainCategory为每篇文档类别所构成的标签向量
def trainNB0(trainMatrix,trainCategory):
    # 获取文档矩阵中文档的数目
    numTrainDocs = len(trainMatrix)
    #print("numTrainDocs:",numTrainDocs)
    #print(trainCategory)
    # 或者文档中一句话的长度,即词条向量的长度
    numWords = len(trainMatrix[0])
    # 文档为侮辱性文档的概率，所有文档中属于类1所占的比例p(c=1)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # 创建一个长度为词条向量等长的列表
    p0Num = ones(numWords); p1Num = ones(numWords)
    # print(p0Num)
    # p0Denom = 0.0; p1Denom = 0.0
    p0Denom = 2.0; p1Denom = 2.0
    # 遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        # 如果该词条向量对应的标签为1
        if trainCategory[i] == 1:
            #print(trainMatrix[i])
            # 统计所有类别为1的词条向量中各个词条出现的次数
            p1Num += trainMatrix[i]
            # 统计类1所有文档中出现单词的数目
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = (p1Num/p1Denom)
    #p0Vect = (p0Num/p0Denom)
    # 采用log函数防止下溢出，通过求对数可以避免下溢出或者浮点数舍入导致的错误
    # 利用NumPy数组计算p(wi|c1)
    p1Vect = log(p1Num / p1Denom)  # change to log()
    # 利用NumPy数组计算p(wi|c0)
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect,p1Vect,pAbusive

# 测试训练算法
trainMat = []
for postInDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postInDoc))
#print(trainMat)
p0V,p1V,pAb = trainNB0(trainMat,listClasses)
#print(pAb)
#print(p1V)
#print(p0V)

# 测试算法：根据现实情况修改分类器
# 将训练函数进行修改以避免0值得影响，修改之前的代码已经注释
# 朴素贝叶斯分类函数：输入要分类的向量以及训练函数trainNB0()返回的三个参数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    # 通过log函数将乘法转换为加法
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# 文档词袋模型，包含单词出现与否以及单词出现次数信息
def bagOfWords2VecMN(vocabList, inputSet):
    # 词袋向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 某词每出现一次，次数加1
            returnVec[vocabList.index(word)] += 1
    return returnVec

