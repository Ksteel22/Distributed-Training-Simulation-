import os
from copy import deepcopy

# import mnist
import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets, utils

import Constent
from Event import Event
from NodeStatus import nodeStatus
from StatisticUtil import statisticUtil
# from WorkNode import WorkNode
from WorkNode_Normal import WorkNode_Normal
# from WorkNode_WithSameData_rand import WorkNode_WithSameData_rand
# from WorkNode_asynchronous import WorkNode_asynchronous
from topological_optimization.Opt_main import Opt_main


class Config:
    def __init__(self, adjacencyMatrix, name):
        self.name = name
        self.nodeBandwidth = 20 * 1e6
        self.nodeFixedDelay = 0.5
        self.adjacencyMatrix = adjacencyMatrix
        self.nodeNum = len(adjacencyMatrix[0])
        self.dataNum = 3000
        self.trainTimes = Constent.TRAIN_TIMES

    def startSimulation(self):
        print(self.nodeNum)

        nodeList = []
        data_root = os.getcwd()
        image_path = data_root + "/DataSet/flower_data/"  # flower data set path
        train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                             transform=Constent.data_transform["train"])

        self.dataNum = len(train_dataset)
        eachNodeData = self.dataNum // self.nodeNum

        train_dataset_divisible = Subset(train_dataset, range(0, self.nodeNum * eachNodeData))
        # data = {"images": mnist.train_images()[:self.dataNum], "labels": mnist.train_labels()[:self.dataNum]}
        for i in range(self.nodeNum):
            print(str(i))
            # data4Node = Subset(train_dataset, range(i * eachNodeData, min(self.dataNum, (i + 1) * eachNodeData)))
            data4Node = torch.utils.data.random_split(train_dataset_divisible, [eachNodeData] * self.nodeNum)
            # data4Node = {"images": mnist.train_images()[i * eachNodeData: min(self.dataNum, (i + 1) * eachNodeData)],
            #              "labels": mnist.train_labels()[i * eachNodeData: min(self.dataNum, (i + 1) * eachNodeData)]}
            neighborSet = set()
            for neighbor in range(self.nodeNum):
                if self.adjacencyMatrix[i][neighbor] == 1:
                    neighborSet.add(neighbor)

            nodeList.append(WorkNode_Normal(i, neighborSet, self.nodeBandwidth, self.nodeFixedDelay, data4Node[i]))

        nodeStatus.initialize(nodeList)
        statisticUtil.initialize(self.nodeNum, self.name)
        print("TEST:::::::::::" + str(statisticUtil.nodeLossList))
        for i in range(self.trainTimes):
            print("*" * 5 + "Epoch " + str(i) + "*" * 5)
            '''
            test event
            '''
            startAllTrainEvent = Event(nodeStatus.startAllTrain, {}, 0, 0)
            startAllTrainEvent.run()

            # nodeStatus.startAllTrain()
            if nodeStatus.isAllFinishTraining():
                nodeStatus.startReceive()

        loss, acc = nodeStatus.printResult()
        statisticUtil.output2Csv()
        print("receiveTime: " + str(statisticUtil.nodeReceiveTime))
        print("trainTime: " + str(statisticUtil.nodeTrainTime))
        return loss, acc


if __name__ == '__main__':
    opt_main = Opt_main(16, 2)
    rrl_A = deepcopy(opt_main.adjacencyMatrix)
    print("rrl_A" + str(rrl_A))
    config = Config(rrl_A, "rrl")
    config.startSimulation()

    # Constent.HAS_PARAMETER = False
    # Constent.PARAMETER = {}
    opt_main.opt()
    opt_A = deepcopy(opt_main.adjacencyMatrix)
    config2 = Config(opt_A, "opt")
    config2.startSimulation()

    clique_A = deepcopy(opt_main.clique())
    config3 = Config(clique_A, "clique")
    config3.startSimulation()

    dep_A = deepcopy(opt_main.dep())
    config4 = Config(dep_A, "dep")
    config4.startSimulation()



