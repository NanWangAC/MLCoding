# -- coding:utf-8 --
from numpy import *
import matplotlib.pyplot as plt

# 加载数据集
def loadDataSet(fileName):
    # 按tab分割特征，并且默认最后一个值是目标值，所以-1
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算最佳拟合直线
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    # 计算X.T*X
    xTx = xMat.T*xMat
    # 判断行列式是否为0，若为0,则矩阵不存在逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 依据公式求取权重，也可利用numpy的库求解，代码为ws = linalg.solve(xTx, xMat.T*yMat)
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 构建权重对角矩阵
    weights = mat(eye((m)))
    print("weight", weights)
    # 随着样本点与待遇测点距离的递增，权重值大小以指数级衰减,参数k衰减速度
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        # 按照高斯核对权重赋值
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    # 判断矩阵可逆性
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 测试局部加权线性回归函数
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def lwlrTestPlot(xArr,yArr,k=1.0):
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

# 计算鲍鱼年龄预测误差
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

# 岭回归：计算回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
    # 按公式构建相应矩阵
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    # 通过求判断行列式的值是否为0来判断矩阵可逆性
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 按公式求解权重.I代表求逆，.T代表求转置
    ws = denom.I * (xMat.T*yMat)
    return ws

# 测试在相应lambda上的回归结果
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    # 在第0个维度上求均值
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    # 在第0个维度上求均值
    xVar = var(xMat,0)
    # 数据标准化，使每维特征具有相同的重要性（不考虑特征代表什么）
    xMat = (xMat - xMeans)/xVar
    # lambda的测试数量
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        # lambda在指数尺度上变化，这样可以看出lambda再取值非常小和非常大情况下对结果造成的影响
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归
# [数据集，预测值，步长，迭代次数]
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    # 数据标准化处理
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat




