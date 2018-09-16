
from numpy import *

# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 以二元方式切分数据集
# [数据集，待切分特征，该特征的某个值]
def binSplitDataSet(dataSet, feature, value):
    # 在第0个维度通过数组过滤方式依据阈值获取相应非零元素的行索引进而将数据集切分成两个子集
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

# 生成叶节点
def regLeaf(dataSet):
    # 返回目标标量的均值
    return mean(dataSet[:,-1])

# 在给定数据集上计算目标变量的平方误差
def regErr(dataSet):
    # 返回总方差
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# 将数据集格式化成目标变量Y和自变量x
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    # 将数据集格式化成目标变量Y和自变量x
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# 生成叶节点
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

# 计算模型误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# 找到数据的最佳二元切分方式
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 容许的误差下降值
    tolS = ops[0]
    # 切分的最少样本数
    tolN = ops[1]
    # 如果所有值相等则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 获得特征和样本数量
    m,n = shape(dataSet)
    # 依据总方差最小来选择切分特征
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果切分后的数据集所含样本点小于阈值则跳过这次循环
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差降低不大于阈值则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分后的数据集小于所允许的最小切分数，则退出函数
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

# 树构建函数
# [数据集，叶节点函数，误差函数，一个包含构建树所需要的其他参数的元组]
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 获得能够最优切分数据集的特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 满足停止条件时返回叶节点的值
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 依据特征和特征值切分数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

# 判断是否为一棵树
def isTree(obj):
    return (type(obj).__name__=='dict')

# 从上往下遍历树，找到两个叶节点计算平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 回归树剪枝函数
# [待剪枝的树，测试集]
def prune(tree, testData):
    # 没有测试数据集则对树进行塌陷处理(即返回树的平均值)
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    # 如果他们不是左右子树，即两者都为叶子则通过计算误差判断是否合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree

# 对回归树叶节点进行预测，为了和modeTreeEval()保持一致，保留两个输入参数
def regTreeEval(model, inData):
    return float(model)

# 对模型树叶节点进行预测，在原数据矩阵上增加第0列，元素的值都是1
def modelTreeEval(model, inData):
    n = shape(inData)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inData
    return float(X*model)

'''
在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
modeEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
调用modelEval()函数，该函数的默认值为regTreeEval()
'''
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

# 以向量的形式返回一组预测值
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

