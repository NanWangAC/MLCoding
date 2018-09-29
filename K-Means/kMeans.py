
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 此处为python3与python2的区别，需要转换为list
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    # 此处python3需要修改，在返回时将数据结构转换为mat
    return mat(dataMat)

# 计算两个向量的欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 产生k*1维的列向量并转换为矩阵存于簇心矩阵中
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# K-均值聚类算法
# [数据集，质心个数，计算距离，创建初始质心]
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据点总数
    m = shape(dataSet)[0]
    # 簇分配结果矩阵包含两列：第一列记录每个点的所属类别，第二列为每个点的存储误差（即点到质心的距离）
    clusterAssment = mat(zeros((m,2)))
    # 获取随机质心集合
    centroids = createCent(dataSet, k)
    # 簇变化标志位
    clusterChanged = True
    # 按照（计算质心-分配-重新计算）反复迭代，只有所有数据点的簇分配结果不在改变
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # 设最小距离为无穷
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算每个数据点到质心的距离，找到每个数据点距离最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        # 更新质心位置
        for cent in range(k):
            # 通过数组过滤来获得给定簇的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            # 沿矩阵列方向进行均值计算
            centroids[cent,:] = mean(ptsInClust, axis=0)
    # 返回质心与分配结果
    return centroids, clusterAssment

# 二分K-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    # 获取数据点总数
    m = shape(dataSet)[0]
    # 存储簇分配结果以及平方误差
    clusterAssment = mat(zeros((m,2)))
    # 创建列表保留所有质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]
    # 存放每个点误差
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    # 不停对簇划分，直到得到想要的簇的数目
    while (len(centList) < k):
        # 初始SSE（误差平方和为无穷大）
        lowestSSE = inf
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit = sum(splitClustAss[:,1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit：{sseSplit}and notSplit:{sseNotSplit} ".format(sseSplit = sseSplit, sseNotSplit = sseNotSplit))
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将新划分后的簇中类别为1的类设为总数据集中最新的一类，类别为0的类设为总数据集中原先的i类
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: {bestCentToSplit}'.format(bestCentToSplit = bestCentToSplit))
        print('the len of bestClustAss is:{LenOfBestClustAss} '.format(LenOfBestClustAss = len(bestClustAss)))
        # 更新质心列表中的变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        # 添加新的类的质心向量
        centList.append(bestNewCents[1,:].tolist()[0])
        # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方和
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    # 返回聚类结果
    return mat(centList), clusterAssment

# 球面距离计算（返回地球表面两点之间的值）
def distSLC(vecA, vecB):
    # 球面余弦定理计算距离，将角度转换为弧度进行计算
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0

# 簇绘图函数
# [希望得到的簇的数目]
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        # 第4列与第五列对应纬度和经度
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

