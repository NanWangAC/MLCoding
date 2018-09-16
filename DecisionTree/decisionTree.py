from math import log
import operator

# 创建一个简单的数据集用于测试
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算给定数据集的熵
def calcShannonEnt(dataSet):
    # 计算数据集中的实例总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 创建数据字典，其键值为最后一列的数值
        currentLabel = featVec[-1]
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 更新标签出现次数
        labelCounts[currentLabel] += 1
    # 初始化熵值
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用类别标签发生的频率计算出现的概率
        prob = float(labelCounts[key])/numEntries
        # 计算熵
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 测试熵计算函数
myDat,labels = createDataSet()
#print(calcShannonEnt(myDat))
myDat[0][-1]='maybe'
#print(myDat)
#print(calcShannonEnt(myDat))

# 按照给定特征划分数据集
'''
@dataSet：待划分的数据集
@axis：划分数据集的特征
@value：需要返回的特征的值
'''
def splitDataSet(dataSet, axis, value):
    # 为了不修改原始数据集，我们创建一个新的列表对象
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        # 发现符合要求的值，将其添加到新创建的列表中
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis + 1:])
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 测试划分数据集函数
#a = [1,2,3]
#b = [4,5,6]
#a.append(b)
#print("append",a)
#c = [7,8,9]
#c.extend(b)
#print("extend",c)
#print(splitDataSet(myDat,1,1))
#print(splitDataSet(myDat,1,0))

# 遍历整个数据集，循环计算熵和splitDataSet()函数，找出最好的特征划分方式
'''
函数中调用的数据要满足一定的要求：数据必须是一种由列表元素组成的列表，而且所有列表元素都要具有相同的数据长度；
数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
'''
def chooseBestFeatureToSplit(dataSet):
    # 有一个长度是标签所以-1
    numFeatures = len(dataSet[0]) - 1
    # 数据集划分之前的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历数据集中所有特征
    for i in range(numFeatures):
        # 将数据集中所有第i个特征值或所有可能存在的值写入新的list中
        featList = [example[i] for example in dataSet]
        # 将list转化为set去掉重复值
        uniqueVals = set(featList)
        # 初始化新的熵值
        newEntropy = 0.0
        # 遍历当前特征中所有唯一属性值，对每个唯一属性值划分一次数据集，得到新的熵值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益，即熵减少情况（指数据无序度的减少情况）
        infoGain = baseEntropy - newEntropy
        # 比较所有特征中的信息增益，返回最好特征划分的索引值
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 测试函数
#print(chooseBestFeatureToSplit(myDat))

# knn分类器，多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树
'''
@dataSet：数据集
@labels：标签列表，包含了数据集中所有特征标签，算法本身不需要这个变量但是为了给出数据的明确含义，我们将他作为一个输入参数提供。
'''
def createTree(dataSet, labels):
    # 将所有数据标签添加到列表中
    classList = [example[-1] for example in dataSet]
    # 如果集合中类别完全相同，则停止划分,即列表中只有该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    '''
    递归函数的第二个停止条件是使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。由于第二个条件无法简单地返回唯一的
    类标签，这里使用majorityCnt()挑选出现次数最多的类别作为返回值
    '''
    #print("dataSet[0]:",dataSet[0])
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的分类属性
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 使用字典类型存储树
    myTree = {bestFeatLabel: {}}
    # 将最优属性从标签中删除，即下次接着从剩下属性中取最优
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    # 将list转化为set去掉重复值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        '''
        得到当前剩余的特征标签,并将类标签复制到subLabels中，python中函数参数使列表类型时，参数使按照引用方式传递的。为了保证每次调用函数createTree()时，
        不改变原始列表内容，使用新变量代替原始列表
        '''
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 测试函数
myTree = createTree(myDat,labels)
# print(myTree)

'''
使用决策树的分类函数
该函数也是一个递归函数，在存储带有数据特征的数据会面临一个问题：程序无法确定特征在数据集中的位置，特征标签列表将帮助程序处理这个问题
@inputTree：决策树模型
@featLabels：标签向量
@testVec：测试样本
'''
def classify(inputTree, featLabels, testVec):
    # 树的第一个键，即根节点
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    print("firstStr：",firstStr)
    # 第一个键对应的值(字典)
    secondDict = inputTree[firstStr]
    print("secondDict：", secondDict)
    # 第一个键(特征)在特征列表中的索引
    featIndex = featLabels.index(firstStr)
    print("featIndex：", featIndex)
    # key是相应特征对应测试列表中的的取值，也即是父子节点间的判断
    key = testVec[featIndex]
    print("key:", key)
    valueOfFeat = secondDict[key]
    print("valueOfFeat:",valueOfFeat)
    # 当valueofFeat不再是字典类型时，退出迭代，此时已经得到分类结果
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
        print("classLabel:",classLabel)
    else:
        classLabel = valueOfFeat
        print("classLabel:", classLabel)
    return classLabel

# 测试函数
print(labels)
print(myTree)
# 因为之前的删除操作，所以此处恢复标签向量
labels = ['no surfacing','flippers']
print(classify(myTree,labels,[1,0]))

# 使用pickle模块存储决策树,pickle模块读写过程中一定要采用二进制读写模式，不然会报错
# 存储模型
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()

# 加载模型
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

