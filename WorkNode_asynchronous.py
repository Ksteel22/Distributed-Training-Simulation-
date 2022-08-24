import time

import numpy as np

import Constent
from Channel import channel
from CnnTrain import CnnTrain
from Node import Node
from NodeStatus import nodeStatus
from Packet import Packet
from StatisticUtil import statisticUtil


class WorkNode_asynchronous(Node):
    def __init__(self, nodeId, neighborList, bandwidth, fixedDelay, data):
        self.nodeId = nodeId
        self.neighborList = neighborList
        self.bandwidth = bandwidth
        self.fixedDelay = fixedDelay
        # self.data = data
        self.simuTime = 0
        self.epoch = Constent.TRAIN_TIMES
        self.tempParamList = {}
        self.lastReceiveTime = 0

        input_len = 13 * 13 * 8
        nodes = 10
        num_filters = 8
        self.cnnTrain = CnnTrain(num_filters, input_len, nodes, data, self.nodeId)
        self.paramList = {"conv_filters": np.random.randn(8, 3, 3) / 9,
                          "softmax_weights": np.random.randn(input_len, nodes) / input_len,
                          "softmax_biases": np.zeros(nodes)}

    def updateParam(self):
        for (key, value) in self.tempParamList.items():
            paramSize = len(value)
            updatedParam = 0
            if key in self.paramList.keys():
                updatedParam = self.paramList.get(key) / (paramSize + 1)

            for param in value:
                updatedParam += param / (paramSize + 1)

            self.paramList[key] = updatedParam

        print("current parameter in Node. " + str(self.nodeId) + "is: " + str(self.paramList))
        self.tempParamList.clear()
        pass

    def collectParam(self, packetParamList):
        for (key, value) in packetParamList.items():
            if key in self.tempParamList.keys():
                self.tempParamList[key].append(value)
            else:
                self.tempParamList[key] = [value]

        pass

    def receivePara(self):
        print("---- node {} start receiving ----".format(self.nodeId))
        packetList = channel.receiveTo(self.nodeId, self.simuTime)
        for packet in packetList:
            self.collectParam(packet.paramList)
        self.updateParam()

        sorted(packetList, key=lambda dataPacket: dataPacket.createTime)
        for packet in packetList:
            # print("receive packet: " + str(packet))
            receiveStartTime = max(self.lastReceiveTime, packet.createTime)

            print("receive start: " + str(receiveStartTime))
            receiveFinishTime = receiveStartTime + packet.packetLength / self.bandwidth + self.fixedDelay
            print("receive finished: " + str(receiveFinishTime) + " :::: using time: " + str(
                receiveFinishTime - receiveStartTime))
            statisticUtil.addReceiveTime(self.nodeId, receiveFinishTime - receiveStartTime)
            self.lastReceiveTime = receiveFinishTime

        packetList.clear()
        self.simuTime = max(self.lastReceiveTime, self.simuTime)
        pass

    def sendPara(self, destinationNodeId):
        """
        - create a packet with source, destination and  parameterList
        - send packet to channel
        :param destinationNodeId:
        :return:
        """
        # print("---- node {} start sending ----".format(self.nodeId))
        packet = Packet(self.nodeId, destinationNodeId, self.paramList, self.simuTime)
        # print("send packet: " + str(packet))
        channel.sendPacket(packet)
        pass

    def train_cnn(self):
        self.cnnTrain.updateParam(self.paramList)
        self.paramList = self.cnnTrain.train_main()

        pass

    def startTrain(self):

        print("=" * 5 + "Epoch " + str(self.epoch) + "=" * 5)
        beginTime = time.time()

        self.train_cnn()

        endTime = time.time()
        # update finish status
        nodeStatus.nodeTrainFinish(self.nodeId)
        # update simuTime
        self.simuTime += endTime - beginTime

        for neighbor in self.neighborList:
            if neighbor is self.nodeId:
                continue

            self.sendPara(neighbor)

        print("train using time: " + str(endTime - beginTime))
        statisticUtil.addTrainTime(self.nodeId, endTime - beginTime)
        print("current time: " + str(self.simuTime))
        self.epoch -= 1

    def printResult(self):
        print("#" * 10)
        print("using time: " + str(self.simuTime))
        return self.cnnTrain.test_cnn()
