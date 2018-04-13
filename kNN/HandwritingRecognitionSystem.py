from numpy import *
from os import listdir
from kNN.KNN import classify0


# 将图像转换为向量
def img2vector(filename):
    # 初始化1*1024的零向量
    returnVect = zeros((1,1024))
    # 打开文件
    fr = open(filename)
    for i in range(32):
        # 读取一行数据，同时每调用一次readline函数，函数指针也会相应移动
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect

# 手写识别系统测试函数
def handwritingClassTest():
    # 初始化手写标签
    hwLabels = []
    # 列出trainingDigits中的文件名
    trainingFileList = listdir('digits/trainingDigits')
    # 获得文件数量
    m = len(trainingFileList)
    # 初始化一个m*1024的训练矩阵，每行数据存储一个图像
    trainingMat = zeros((m,1024))
    for i in range(m):
        # 按照文件名格式分解处理，文件命名规则如下：9_45.txt的分类是9，它是数字9的第45个实例
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("digits/trainingDigits/%s"%fileNameStr)
    testFileList = listdir('digits/testDigits/')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("digits/testDigits/%s"%fileNameStr)
        classifierResult = classify0( vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is %d"%(classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total number of errors is: %d"%errorCount)
    print("\n the total error rate is: %f"%(errorCount/float(mTest)))

# 测试img2vector
testVector = img2vector("digits/testDigits/0_13.txt")
print(testVector[0,0:31])

# 测试 handwritingClassTest
handwritingClassTest()