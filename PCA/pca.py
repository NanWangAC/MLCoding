
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # python3应在map函数后加list()
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

# PCA算法
# [数据集，应用特征数]
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    # 去除平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=False)
    # 计算协方差矩阵的特征值及对应的特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    # 对特征值进行从小到大排序
    eigValInd = argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:,eigValInd]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat, reconMat

# 绘制通过PCA降维前后的数据分布
def compareOriginalAndPCA(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flatten()将矩阵化为仅含一个元素的数组
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^' , s=90, c='blue')
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o' , s=50, c='red')
    plt.show()

# 将NaN(not a number 即缺失值)替换为平均值
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 利用该维度所有非NaN特征求取均值，通过isnan函数得到矩阵（数组）中缺失值的情况，再取反（~操作）得到非缺失值的索引
        # 然后对非缺失值求平均
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        # 将该维度中所有NaN特征全部用均值替换
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat


