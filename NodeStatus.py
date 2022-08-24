class NodeStatus:
    def __init__(self):
        self.nodeList = []
        self.nodeTrainBook = []
        pass

    def isAllFinishTraining(self):
        """
        if all nodes finish training -> return true
        else -> return false
        :return: true/false
        """
        for i in self.nodeTrainBook:
            if i == 0:
                return False
        self.nodeTrainBook = [0] * len(self.nodeTrainBook)
        return True

    def startAllTrain(self, parameters):
        """
        control all nodes to start train
        """
        for node in self.nodeList:
            node.startTrain()
        pass

    def startReceive(self):
        """
        control all nodes to receive packets from channel
        :return:
        """
        for node in self.nodeList:
            node.receivePara()
        pass

    def initialize(self, nodeList):
        """
        initialize this NodeStatus
        :param nodeList:
        :return:
        """
        self.nodeList = nodeList
        self.nodeTrainBook = [0] * len(nodeList)
        for node in self.nodeList:
            node.initParam2Cnn()

    def nodeTrainFinish(self, nodeId):
        """
        this node id packet finish train
        :param nodeId:
        :return:
        """
        print(self.nodeTrainBook)
        print(nodeId)
        self.nodeTrainBook[nodeId] = 1

    def printResult(self):
        loss = 0
        acc = 0
        for node in self.nodeList:
            nodeLoss, nodeAcc = node.printResult()
            loss += nodeLoss / len(self.nodeList)
            acc += nodeAcc / len(self.nodeList)

        return loss, acc


nodeStatus = NodeStatus()
