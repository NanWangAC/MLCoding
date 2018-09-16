
from numpy import *

# 构建简单的数据
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    # 将数组元素全部设置为1
    retArray = ones((shape(dataMatrix)[0],1))
    # 将所有不满足不等式要求的元素设置为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    return retArray
    
# 遍历stumpClassify()函数所有可能的输入值，，在一个加权数据集中循环，并找到数据集上最佳的单层决策树，D为权重
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # 该变量用于在特征所有可能的值上进行遍历
    numSteps = 10.0
    # 构建空字典，用于存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    # 将最小错误率初始化成无穷大
    minError = inf
    # 在数据集所有特征上遍历
    for i in range(n):
        # 计算最大值与最小值
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        # 求取步长
        stepSize = (rangeMax-rangeMin)/numSteps
        # 对特征所有的可能值进行遍历
        for j in range(-1,int(numSteps)+1):
            # 对每个不等号进行遍历
            # lt:小于 gt:大于
            for inequal in ['lt', 'gt']:
                # 设立阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 建立一颗单层决策树并利用加权数据集对他进行测试
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                # 建立错误数组判断预测情况
                errArr = mat(ones((m,1)))
                # 预测正确时，相应的位置为1
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率
                weightedError = D.T*errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
                if weightedError < minError:
                    minError = weightedError
                    # 因为python会通过引用的方式传递所有列表，所以必须明确告知python要为bestClasEst分配新内存
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

# 基于单层决策树的AdaBoost训练过程
# [数据集，标签，迭代次数=40]
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    # 存储分类信息
    weakClassArr = []
    m = shape(dataArr)[0]
    # D为概率分布向量，因此其所有元素之和唯一，初始化时使每个元素概率均等为1/m
    D = mat(ones((m,1))/m)
    # 用于记录每个点的类别估计值
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        # 建立单层决策树，同时得到相应的最小错误率和估计的类别向量
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        # 计算alpha值，alpha值会告诉总分类器本次单层决策树输出结果的权重，其中max(error,le-16)用于确保不会发生除0溢出
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        # 更新权重
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        # 错误率累加计算
        aggClassEst += alpha*classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

# AdaBoost分类函数
# [待分类样例，弱分类器数组]
def adaClassify(datToClass,classifierArr):
    # 将待分类样例转换成numpy矩阵
    dataMatrix = mat(datToClass)
    # 获取待分类样例个数m
    m = shape(dataMatrix)[0]
    # 记录每个点的类别估计值
    aggClassEst = mat(zeros((m,1)))
    # 遍历弱分类器
    for i in range(len(classifierArr)):
        # 对每一个分类器得到一个类别估计值
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 将类别估计值乘上alpha权重然后累计得到每个点的类别估计值
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

# ROC曲线回执以及AUC函数的计算
# [分类器的预测强度，分类标签]
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    # 绘制光标的位置
    cur = (1.0,1.0)
    # ySum用于计算AUC的值
    ySum = 0.0
    # 通过数组过滤的方式计算正例的数目
    numPosClas = sum(array(classLabels)==1.0)
    # 确定坐标绘制步长
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    # 获得排序索引，从小到大排序，也就意味着负例在前，正例在后
    print("predStrengths:", predStrengths)
    sortedIndicies = predStrengths.argsort()
    print("sortedIndicies", sortedIndicies)
    # 构建画笔
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 将排序值转换为列表，然后在列表上进行循环迭代
    for index in sortedIndicies.tolist()[0]:
        # 每得到一个标签为1.0的类，就沿着y轴下降一个步长，即不断降低真阳率（真正例的比例）
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        # 得到其他标签时，则在x轴倒退一个步长（假阳率方向）
        else:
            delX = xStep; delY = 0
            # 对小矩形的高度进行累加
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

