from numpy import *

# 加载数据
def loadDataSet():
    # 创建初级与标签列表
    dataMat = []; labelMat = []
    # 打开文本数据集
    fr = open('testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行除去首尾空格之后按空格进行分离
        lineArr = line.strip().split()
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    # 返回数据列表，标签列表
    return dataMat,labelMat

#定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升法更新最优拟合参数
#@dataMatIn：数据集
#@classLabels：数据标签
def gradAscent(dataMatIn, classLabels):
    # 将数据集列表转为Numpy矩阵
    dataMatrix = mat(dataMatIn)
    # 将数据集标签列表转为Numpy矩阵，并转置
    labelMat = mat(classLabels).transpose()
    # 获取数据集矩阵的行数和列数
    m,n = shape(dataMatrix)
    # 学习率
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 初始化权值参数向量每个维度均为1.0
    weights = ones((n,1))
    # 循环迭代以及向量化计算
    for k in range(maxCycles):
        # 求当前的sigmoid函数预测概率
        h = sigmoid(dataMatrix*weights)
        # 计算真实类别和预测类别的差值
        error = (labelMat - h)
        # 更新权值参数，更新公式通过求导得到
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights

# 画出数据集合logistics回归最佳拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    # 获取数据集的行数和列数
    m,n = shape(dataMatrix)
    # 设置学习率为0.01
    alpha = 0.01
    # 初始化权值向量各个参数为1.0
    weights = ones(n)
    # 循环m次，每次选取数据集一个样本更新参数
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
#@dataMatrix：数据集列表
#@classLabels：标签列表
#@numIter：迭代次数，默认150
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 获取数据集的行数和列数
    m,n = shape(dataMatrix)
    # 初始化权值参数向量每个维度均为1
    weights = ones(n)
    # 循环每次迭代次数
    for j in range(numIter):
        # 获取数据集行下标列表
        dataIndex = list(range(m))
        # 遍历行列表
        for i in range(m):
            # 每次更新参数时设置动态的步长，保证随着更新次数的增多，步长变小，避免在最小值处徘徊
            alpha = 4/(1.0+j+i)+0.0001
            # 随机获取样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            # 计算权值更新
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del(dataIndex[randIndex])
    return weights

# 分类决策函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#logistic回归预测算法
def colicTest():
    # 打开训练数据集
    frTrain = open('horseColicTraining.txt')
    # 打开测试数据集
    frTest = open('horseColicTest.txt')
    # 新建两个孔列表，用于保存训练数据集和标签
    trainingSet = []; trainingLabels = []
    # 读取训练集文档的每一行
    for line in frTrain.readlines():
        # 对当前行进行特征分割
        currLine = line.strip().split('\t')
        # 新建列表存储每个样本的特征向量
        lineArr =[]
        for i in range(21):
            # 将该样本的特征存入lineArr列表
            lineArr.append(float(currLine[i]))
        # 将该样本的特征向量添加到数据集列表
        trainingSet.append(lineArr)
        # 将该样本标签存入标签列表
        trainingLabels.append(float(currLine[21]))
    # 调用随机梯度上升法更新logistic回归的权值参数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 5000)
    # 统计测试数据集预测错误样本数量和样本总数
    errorCount = 0; numTestVec = 0.0
    # 遍历测试数据集的每个样本
    for line in frTest.readlines():
        # 样本总数加1
        numTestVec += 1.0
        # 对当前行进行处理，分割出各个特征及样本标签
        currLine = line.strip().split('\t')
        # 新建特征向量
        lineArr =[]
        # 将各个特征构成特征向量
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            # 如果预测错误，错误数加1
            errorCount += 1
    # 计算测试集总的预测错误率
    errorRate = (float(errorCount)/numTestVec)
    # 打印错误率大小
    print("the error rate of this test is: %f" % errorRate)
    # 返回错误率
    return errorRate

#多次测试算法求取预测误差平均值
def multiTest():
    # 设置测试次数为10次，并统计错误率总和
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    # 打印出测试10次预测错误率平均值
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

# 测试loadDataSet()/sigmoid(inx)/gradAscent(dataMatIn,classLabels)
# dataArr,labelMat = loadDataSet()
# weight = gradAscent(dataArr,labelMat)
# print(weight)
# # 测试画线函数
# # getA()函数与mat()函数的功能相反，是将一个numpy矩阵转换为数组
# plotBestFit(weight.getA())
# # 测试随机梯度上升
# weight = stocGradAscent0(array(dataArr),labelMat)
# plotBestFit(weight)
# # 测试改进的随机梯度上升
# weight = stocGradAscent1(array(dataArr),labelMat)
# plotBestFit(weight)
# 进行分类
multiTest()