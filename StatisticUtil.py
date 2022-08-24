from Out2Csv import Out2Csv

# import matplotlib.pyplot as plt


class StatisticUtil:
    def __init__(self):
        self.nodeLossList = []
        self.nodeAccList = []
        self.nodeTrainTime = []
        self.nodeReceiveTime = []
        self.nodeNum = 0
        self.name = " "

    def addAcc(self, nodeId, acc):
        self.nodeAccList[nodeId].append(acc)

    def addLoss(self, nodeId, loss):
        self.nodeLossList[nodeId].append(loss)

    def addTrainTime(self, nodeId, trainTime):
        self.nodeTrainTime[nodeId] += trainTime
        pass

    def addReceiveTime(self, nodeId, receiveTime):
        self.nodeReceiveTime[nodeId] += receiveTime
        pass

    def output2Csv(self):
        outLoss2Csv = Out2Csv("lossEachEpoch" + self.name + ".csv")
        outAcc2Csv = Out2Csv("accEachEpoch" + self.name + ".csv")
        outAvgLoss = Out2Csv("avgLoss" + self.name + ".csv")
        outAvgAcc = Out2Csv("avgAcc" + self.name + ".csv")

        for j in range(len(self.nodeAccList[0])):
            sumLoss = 0
            sunAcc = 0
            for i in range(self.nodeNum):
                sumLoss += self.nodeLossList[i][j]
                sunAcc += self.nodeAccList[i][j]

            outAvgAcc.addPoint(str(sunAcc / self.nodeNum))
            outAvgLoss.addPoint(str(sumLoss / self.nodeNum))

        for i in range(self.nodeNum):
            strEpoch_acc = str(i) + ","
            for data in self.nodeAccList[i]:
                strEpoch_acc += str(data) + ","
            outAcc2Csv.addPoint(strEpoch_acc)

            strEpoch_loss = str(i) + ","
            for data in self.nodeLossList[i]:
                strEpoch_loss += str(data) + ","
            outLoss2Csv.addPoint(strEpoch_loss)

            # self.drawPic(i)

        pass

    def initialize(self, nodeNum, name):
        self.nodeNum = nodeNum
        self.nodeLossList.clear()
        self.nodeAccList.clear()
        self.nodeTrainTime = [0] * nodeNum
        self.nodeReceiveTime = [0] * nodeNum
        self.name = name
        for i in range(nodeNum):
            self.nodeLossList.append([])
            self.nodeAccList.append([])

    def drawPic(self, nodeId):
        # plt.plot(range(len(self.nodeAccList[nodeId])), self.nodeAccList[nodeId])
        # plt.plot(range(len(self.nodeLossList[nodeId])), self.nodeLossList[nodeId])
        # plt.show()
        pass


statisticUtil = StatisticUtil()
