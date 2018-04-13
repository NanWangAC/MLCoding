from numpy import *
import operator
import matplotlib.pyplot as plt

# 创造简单的数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# k-近邻算法
def classify0(inX, dataSet, labels, k):
    # 读取矩阵第一维度的长度,即获取行数
    dataSize = dataSet.shape[0]
    # 将inX扩展为与dataset一样的矩阵然后与dataset进行相减获得相对坐标值为求距离做准备
    diffMat = tile(inX, (dataSize,1)) - dataSet
    # 将相对坐标值求乘方，即（x1-x2）^2
    sqDiffMat = diffMat**2
    # axos = 0代表将每列之和相加，axis = 1代表将每行之和相加，即得到了（x1-x2）^2+(y1-y2)^2
    sqDistances = sqDiffMat.sum(axis = 1)
    # 将和开方即得到欧氏距离
    distances = sqDistances**0.5
    # argsort是排序，将元素按照由小到大的顺序返回下标
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        # 按从近到远的方式获得相应的类别标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # 对标签进行累加计数，若标签已存在则继续累加值，不存在则更新
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 遍历classCount这一字典中所有的元组，然后按照value进行降序排列，reverse = True即为降序排列，key=operator.itemgetter(1)即按照元组的第二个元素进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    # 返回类别最多的类
    return  sortedClassCount[0][0]

'''
DEMO：通过kNN改进约会网站匹配效果
'''

# 将文本记录转换为NumPy的解析程序
def file2matrux(filename):
    # 打开文件
    fr = open(filename)
    # 逐行读取文件
    arrayOLines = fr.readlines()
    # 获取文件的行数
    numberOfLines = len(arrayOLines)
    # 创建一个numberOfLines行，3列的0矩阵
    returnMat = zeros((numberOfLines,3))
    # 定义一个空数组
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 去除首尾空格以及各类换行符
        line = line.strip()
        # 通过指定分割符对字符串切分
        listFromLine = line.split('\t')
        # 将数据更新入矩阵相应的行数中
        returnMat[index,:] = listFromLine[0:3]
        # 将listFromLine倒数第一个元素转化为整型并添加到向量中
        classLabelVector.append(int(listFromLine[-1]))
        # 索引自增
        index += 1
    return returnMat,classLabelVector

#


'''
归一化特征值（处理同等重要性但是不同取值范围的特征值的常用方法）
下面的公式可以将任意取值范围的特征值转化为0-1区间内的值
newValue = (oldValue - min)/(max - min)
'''
def autoNorm(dataSet):
    # 参数0使得函数可以从列中选取最小值而不是选取当前行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 分类器准确度测试
def datingClassTest():
    # 百分之十的数据用于测试分类器 更改该变量的值可更改参加测试分类器的数据量
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrux('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 分类器（需要测试的向量，训练样本集(90%)，标签集合，K）
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f"%(errorCount/float(numTestVecs)))

# 分类结果预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrux('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will propably like this person: ",resultList[classifierResult - 1])


# 数据可视化
# 建立一个画板
fig = plt.figure()
# 添加子图，将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax = fig.add_subplot(111)
datingDataMat,datingLabels = file2matrux('datingTestSet2.txt')
# 绘制散点图,横坐标取矩阵第2列所有数据，纵坐标取矩阵第三列数据,绘制色彩不同，尺寸不同的点
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# 显示
plt.show()


group,lables = createDataSet()
print(group,lables)
print("===============================================")
print(classify0([0,0],group,lables,3))
print("===============================================")
print(datingDataMat)
print(datingLabels[0:20])
print("===============================================")
normMat,ranges,minVals = autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)
print("===============================================")
print(datingClassTest())
print("===============================================")
classifyPerson()
