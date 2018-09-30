
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 基于欧几里得距离计算相似度
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 计算pearson相关系数
def pearsSim(inA,inB):
    if len(inA) < 3 :
        return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = False)[0][1]

# 计算余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

# 基于物品相似度计算评分值
# [数据矩阵，用户编号，相似度计算，物品编号]
def standEst(dataMat, user, simMeas, item):
    # 计算列的数量，即物品的数量
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        # 如果用户没有对物品j进行打分，那么跳过
        if userRating == 0:
            continue
        # 找到两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # 利用相似度计算两个物品之间的相似度
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        print('the {item} and {j} similarity is: {similarity}'.format (item=item, j=j, similarity=similarity))
        # 累加相似度
        simTotal += similarity
        # 累加待推荐物品与用户打过分的物品之间的相似度*用户对物品的打分
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 对相似度的评分乘积进行归一化
        return ratSimTotal/simTotal

# 基于SVD评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U,Sigma,VT = la.svd(dataMat)
    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵，此处采用包含了90%的能量4个主要特征
    Sig4 = mat(eye(4)*Sigma[:4])
    print("sigma",Sigma)
    print("sig",Sig4)
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the {item} and {j} similarity is: {similarity}'.format(item=item, j=j, similarity=similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

# 返回最高的N个推荐结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找用户未评级的物品，并建立未评分物品列表
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    # 若不存在未评分物品则退出
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        # 获得物品预测评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 根据评级逆序排列（从大到小），然后返回N个评级最高的物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# 打印矩阵
# [数据矩阵，阈值]
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            # 由于矩阵中包含浮点数，所以采用阈值区分浅色和灰色
            if float(inMat[i,k]) > thresh:
                print (1,end='')
            else:
                print (0,end='')
        print ('')

# 图像压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using {numSV} singular values******".format(numSV=numSV))
    printMat(reconMat, thresh)

imgCompress(2)