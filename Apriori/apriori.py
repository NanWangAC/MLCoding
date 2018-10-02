
from numpy import *

# 简单的测试数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 构建一个大小为1的所有候选项的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 这里不是简单的添加物品项目，而是添加只包含该物品项的一个列表
                C1.append([item])
    # 将C1从小到大排序
    C1.sort()
    # 对C1中每个项构建一个不变的集合
    return list(map(frozenset, C1))

# 将满足最小支持度的集合构成项集
# [数据集，候选集列表，最小支持度]
def scanD(dataSet, candidateK, minSupport):
    ssCnt = {}
    for tid in dataSet:
        for can in candidateK:
            # 判断can是不是tid的子集
            if can.issubset(tid):
                # 累计can的出现次数
                if not can in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can] += 1
    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算所有项集的支持度
        support = ssCnt[key]/numItems
        # 将满足最小支持度的项添加到retlist中
        if support >= minSupport:
            retList.insert(0,key)
        # 存放相应项支持度备用
        supportData[key] = support
    return retList, supportData

# 创建候选集
# [频繁项集列表，项集中每项（子集）的元素个数]
def aprioriGenerate(frequentItemSetk, k):
    retList = []
    for i in range(len(frequentItemSetk)):
        # 当前K-2个项相同时，将两个集合合并
        for j in range(i+1, len(frequentItemSetk)):
            L1 = list(frequentItemSetk[i])[:k - 2]
            L2 = list(frequentItemSetk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(frequentItemSetk[i] | frequentItemSetk[j])
    return retList

# Apriori算法
# [数据集，最小支持度]
def apriori(dataSet, minSupport = 0.5):
    candidate1 = createC1(dataSet)
    dataSet = list(map(set, dataSet))
    # 构建初始的频繁项集，即所有项集只有一个元素
    frequentItemSet1, supportData = scanD(dataSet, candidate1, minSupport)
    frequentItemSet = [frequentItemSet1]
    # 最初的L1中的每个项集含有一个元素，新生成的项集应该含有2个元素，所以 k=2
    k = 2
    # 从含有i个元素的项集遍历
    i = 0
    while (len(frequentItemSet[i]) > 0):
        candidateK = aprioriGenerate(frequentItemSet[i], k)
        frequentItemSetK, supK = scanD(dataSet, candidateK, minSupport)
        # 将新的项集的支持度数据加入原来的总支持度字典中
        supportData.update(supK)
        # 将符合最小支持度要求的项集加入L
        frequentItemSet.append(frequentItemSetK)
        # 新生成的项集中的元素个数应不断增加
        k += 1
        i += 1
    # 返回所有满足条件的频繁项集的列表，和所有候选项集的支持度信息
    return frequentItemSet, supportData

# 生成关联规则
# [频繁项集，包含频繁项集支持数据的字典，最小可信度阈值]
def generateRules(frequentItemSet, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(frequentItemSet)):
        for frequentItemSet in frequentItemSet[i]:
            # 遍历频繁项集集合中的每一个频繁项集然后创建只包含单个元素的列表
            H1 = [frozenset([item]) for item in frequentItemSet]
            # 如果频繁项集的元素数目超过2，则对它进行进一步合并，若只有两个元素则计算置信度
            if (i > 1):
                rulesFromConsequence(frequentItemSet, H1, supportData, bigRuleList, minConf)
            else:
                calculateConfidence(frequentItemSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

# 对规则进行评估
def calculateConfidence(frequentItemSet, H, supportData, bigRuleList, minConf=0.7):
    # 存放满足最小可信度的规则
    prunedH = []
    # 遍历H计算置信度
    for consequence in H:
        # 使用supportData中的支持度数据计算可信度可以节省大量时间
        confidence = supportData[frequentItemSet] / supportData[frequentItemSet - consequence]
        if confidence >= minConf:
            print(frequentItemSet - consequence, '-->', consequence, 'confidence:', confidence)
            bigRuleList.append((frequentItemSet - consequence, consequence, confidence))
            prunedH.append(consequence)
    return prunedH

# 生成候选集合规则
def rulesFromConsequence(frequentItemSet, H, supportData, bigRuleList, minConf=0.7):
    m = len(H[0])
    if (len(frequentItemSet) > (m + 1)):
        # 生成无重复组合存放在Hmp1,里头有所有可能的规则
        Hmp1 = aprioriGenerate(H, m + 1)
        # 通过计算可信度来过滤得到符合要求的规则
        Hmp1 = calculateConfidence(frequentItemSet, Hmp1, supportData, bigRuleList, minConf)
        # 如果不止一条规则，则尝试进一步组织这些规则
        if (len(Hmp1) > 1):
            rulesFromConsequence(frequentItemSet, Hmp1, supportData, bigRuleList, minConf)
