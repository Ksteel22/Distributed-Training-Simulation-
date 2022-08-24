import copy
import time
from random import random

import numpy as np

import Constent
from Channel import channel
# from CnnTrain import CnnTrain
from Node import Node
from NodeStatus import nodeStatus
from Packet import Packet
from StatisticUtil import statisticUtil
from TrainModel_AlexNet import AlexNetTrain


class WorkNode_Normal(Node):
    def __init__(self, nodeId, neighborList, bandwidth, fixedDelay, data):
        self.nodeId = nodeId
        self.neighborList = neighborList
        self.bandwidth = bandwidth
        self.fixedDelay = fixedDelay
        # self.data = data
        self.simuTime = 0
        self.tempParamList = {}
        self.lastReceiveTime = 0

        self.model = AlexNetTrain(data, self.nodeId)
        # input_len = 13 * 13 * 8
        # nodes = 10
        # num_filters = 8
        # self.cnnTrain = CnnTrain(num_filters, input_len, nodes, data, self.nodeId)
        if Constent.HAS_PARAMETER:
            self.paramList = copy.deepcopy(Constent.PARAMETER)
        else:
            self.paramList = self.model.getParam()
            Constent.HAS_PARAMETER = True
            Constent.PARAMETER = copy.deepcopy(self.paramList)

    # def initParam2Cnn(self):
    #     self.cnnTrain.updateParam(self.paramList)

    def updateParam(self):
        for key in self.tempParamList.keys():
            paramSize = len(self.tempParamList.get(key))
            # updatedParam = 0
            # if key in self.paramList.keys():
            updatedParam = self.paramList.get(key) / (paramSize + 1)
            print("node {} update param {}".format(self.nodeId, key))
            for param in self.tempParamList.get(key):
                # print("before: " + str(self.paramList.get(key)[0]))

                addParam = param
                # print("add " + str(addParam[0]))
                updatedParam = updatedParam + param / (paramSize + 1)
                # print("after: " + str(updatedParam[0]))
                # print("after: " + str(updatedParam[0]))

            self.paramList[key] = updatedParam

        # print("current parameter in Node. " + str(self.nodeId) + "is: " + str(self.paramList))

        self.tempParamList.clear()
        self.initParam2Cnn()
        pass

    def collectParam(self, packetParamList):
        for (key, value) in packetParamList.items():
            if key in self.tempParamList.keys():
                self.tempParamList[key].append(value)
            else:
                self.tempParamList[key] = [value]

        pass

    def initParam2Cnn(self):
        pass

    def receivePara(self):
        """
        1. receive packets from neighbors
        2. update related parameters
        3. calculate translation time and update simuTime
        """
        # print("---- node {} start receiving ----".format(self.nodeId))
        # 1.receive packets from neighbors
        packetList = channel.receiveTo(self.nodeId, float("inf"))

        # 2. update related parameters
        for packet in packetList:
            # print("{} self: {}".format(self.nodeId, self.paramList["softmax_biases"][0]))
            # print("{} receive from {}: {}".format(self.nodeId, packet.sourceNodeId, packet.paramList["softmax_biases"][0]))
            self.collectParam(packet.paramList)
        self.updateParam()
        # 3. calculate translation time and update simuTime
        sorted(packetList, key=lambda dataPacket: dataPacket.createTime)
        # sumTime = 0
        # print("receive packets num: " + str(len(packetList)))
        for packet in packetList:
            # print("receive packet from: " + str(packet.sourceNodeId))
            receiveStartTime = max(self.lastReceiveTime, packet.createTime)

            # print("receive start: " + str(receiveStartTime))
            receiveFinishTime = receiveStartTime + packet.packetLength / self.bandwidth + self.fixedDelay
            # print("receive finished: " + str(receiveFinishTime) + " :::: using time: " + str(
            #     receiveFinishTime - receiveStartTime))
            statisticUtil.addReceiveTime(self.nodeId, receiveFinishTime - receiveStartTime)
            self.lastReceiveTime = receiveFinishTime

        packetList.clear()
        # current simulation time is time after train
        # lastReceiveTime is time after receive
        self.simuTime = max(self.lastReceiveTime, self.simuTime)

        print("current time: " + str(self.simuTime))
        pass

    def sendPara(self, destinationNodeId):
        """
        - create a packet with source, destination and  parameterList
        - send packet to channel
        :param destinationNodeId:
        :return:
        """
        # print("---- node {} start sending ----".format(self.nodeId))
        packet = Packet(self.nodeId, destinationNodeId, copy.deepcopy(self.paramList), copy.deepcopy(self.simuTime))
        # print("{} send packet to {}: {}".format(self.nodeId, destinationNodeId, self.paramList["softmax_biases"][0]))
        channel.sendPacket(copy.deepcopy(packet))
        pass

    def train_test(self):
        """
        create a random num as parameter
        :return:
        """
        print("---- node {} start training ----".format(self.nodeId))
        time.sleep(0.05)
        self.tempParamList.clear()
        if "num" not in self.paramList:
            self.paramList["num"] = 10 * random()

    def train_cnn(self):
        self.cnnTrain.updateParam(self.paramList)
        # train and get parameter list from cnnTrain
        self.paramList = self.cnnTrain.train_main()

        pass

    def startTrain(self):
        print("node {} start train".format(self.nodeId))
        self.model.updateParam(self.paramList)
        beginTime = time.time()
        # self.train_cnn()
        self.paramList = self.model.train_main()
        endTime = time.time()

        # update simuTime
        self.simuTime += endTime - beginTime

        print("node {} start transmit".format(self.nodeId))
        for neighbor in self.neighborList:
            if neighbor is self.nodeId:
                continue
            self.sendPara(neighbor)

        # update finish status
        nodeStatus.nodeTrainFinish(self.nodeId)
        print("train using time: " + str(endTime - beginTime))

        statisticUtil.addTrainTime(self.nodeId, endTime - beginTime)
        print("current time: " + str(self.simuTime))

    def printResult(self):
        print("#" * 10)
        print("using time: " + str(self.simuTime))
        return self.model.test()
