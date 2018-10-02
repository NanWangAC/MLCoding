
# FP-树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        # 链接相似的元素项
        self.nodeLink = None
        self.parent = parentNode
        self.children = {} 

    def counter(self, numOccur):
        self.count += numOccur

    # 将树以文本形式显示
    def display(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)

# 使用数据集以及最小支持度作为参数来构建FP树
def createTree(dataSet, minSup=1):
    headerTable = {}
    # 遍历数据集并统计每个元素项出现的频度
    for trans in dataSet:
        for item in trans:
            # 累加出现次数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable):
        if headerTable[k] < minSup: 
            del(headerTable[k])
    frequentItemSet = set(headerTable.keys())
    # 如果不存在频繁项集则不需要进行下一步处理
    if len(frequentItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 初始化FP-树，创建根节点
    retTree = treeNode('Null Set', 1, None)
    # 第二次遍历数据
    for tranSet, count in dataSet.items():
        localData = {}
        for item in tranSet:
            if item in frequentItemSet:
                # 第二次遍历中只对频繁项集进行操作
                localData[item] = headerTable[item][0]
        if len(localData) > 0:
            # 根据全局频率对每个事务中的元素从大到小进行排列
            orderedItems = [v[0] for v in sorted(localData.items(), key=lambda p: p[1], reverse=True)]
            # 使用排序后的频率项集对树进行填充（也就是使其生长，即growth）
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

# 根据数据更新树
def updateTree(items, inTree, headerTable, count):
    # 检查是否存在该节点
    if items[0] in inTree.children:
        # 存在则计数增加
        inTree.children[items[0]].counter(count)
    else:
        # 不存在则建新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            # 如果该字母首次出现，则直接将字母表的头指针指向该结点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 否则，需要将其插入到合适的位置，此处为尾插法
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    # 若仍有未分配完的树，继续迭代
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

# 确保节点链接指向树中该元素项的每一个实例
def updateHeader(nodeToTest, targetNode):
    # 找到表尾
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    # 使指针指向表尾
    nodeToTest.nodeLink = targetNode

# 迭代上溯整棵树
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

# 获得条件模式基（以所查找元素为结尾的路径集合）
def findPrefixPath(treeNode):
    # 存放条件模式基
    conditionPatternsBase = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            conditionPatternsBase[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return conditionPatternsBase

# 从FP树中挖掘频繁项集
def mineTree(headerTable, minSup, preFix, freqItemList):
    # 对头指针表中的元素项按照出现频率从小到大进行排序，即从表底端进行操作
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        # 获取条件模式基
        conditionPatternsBase = findPrefixPath(headerTable[basePat][1])
        # 利用条件模式基创建FP树
        myCondTree, myHead = createTree(conditionPatternsBase, minSup)
        # 若树中有元素项，则递归
        if myHead != None:
            print('conditional tree for:',newFreqSet)
            myCondTree.display(1)
            mineTree(myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpleData = loadSimpDat()
initSet = createInitSet(simpleData)
myFPTree, myHeaderTab = createTree(initSet, 3)
print(myHeaderTab)

freqItems = []
mineTree(myHeaderTab, 3, set([]), freqItems)