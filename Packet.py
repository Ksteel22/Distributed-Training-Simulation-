from sys import getsizeof


class Packet:
    @staticmethod
    def calculateLength(paramList):
        sumLength = 0
        for param in paramList:
            sumLength += getsizeof(param)

        return sumLength

    def __init__(self, sourceNodeId, destinationNodeId, paramList, createTime):
        self.sourceNodeId = sourceNodeId
        self.destinationNodeId = destinationNodeId
        self.paramList = paramList
        self.createTime = createTime
        self.packetLength = self.calculateLength(paramList)

        pass

    def __str__(self):
        return "Packet:[sourceNodeId: {}, destinationNodeId: {}, paramList: {}, createTime: {}, packetLength: {}]"\
            .format(self.sourceNodeId, self.destinationNodeId, self.paramList, self.createTime, self.packetLength)


