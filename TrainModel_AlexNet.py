import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from torchvision import transforms, datasets, utils

import numpy as np

import Constent
from AlexNet.model import AlexNet
from StatisticUtil import statisticUtil
from TrainModel import TrainModel


class AlexNetTrain(TrainModel):
    def __init__(self, dataset, fromNode):
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.net = AlexNet(num_classes=5, init_weights=True)
        self.net.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.net.parameters(), lr=0.0002)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.002)
        self.save_path = './AlexNet.pth'
        self.best_acc = 0.0

        self.dataset = dataset
        self.fromNodeId = fromNode

        pass

    def updateParam(self, paramList):
        self.net.load_state_dict(paramList)
        pass

    def getParam(self):
        return self.net.state_dict()

    def test(self):
        data_root = os.getcwd()
        dataPath = data_root + "/DataSet/flower_data/"
        validate_dataset = datasets.ImageFolder(root=dataPath + "/val",
                                                transform=Constent.data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=1, shuffle=True,
                                                      num_workers=0)

        self.net.eval()  
        acc = 0.0  # accumulate accurate number / epoch
        test_loss = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = self.net(val_images.to(self.device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(self.device)).sum().item()

                loss = self.loss_function(outputs, val_labels.to(self.device))
                test_loss += loss.item()
            val_accurate = acc / val_num
            # if val_accurate > self.best_acc:
            #     best_acc = val_accurate
            #     torch.save(self.net.state_dict(), self.save_path)
        print("test: " + str(test_loss / val_num))
        return test_loss / val_num, val_accurate

    def train_main(self):
        # permutation = np.random.permutation(len(self.dataset))
        parameterList = self.getParam()

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=1, shuffle=True)

        self.net.train()
        # print(next(self.net.parameters()).is_cuda)
        running_loss = 0.0
        running_corrects = 0.0
        for i, data in enumerate(train_loader):
            im, label = data
            im = im.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(im)
            _, predicts = torch.max(outputs, 1)

            loss = self.loss_function(outputs, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            # print(predicts.is_cuda, label.is_cuda)
            running_corrects += torch.sum(predicts == label.data).item()

            rate = (i + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f},{:.3f}".format(int(rate * 100), a, b,
                                                                        running_loss / len(train_loader),
                                                                        running_corrects / len(train_loader)), end="")

        print()
        # self.net.eval()
        # test_loss, acc = self.test()
        statisticUtil.addAcc(self.fromNodeId, running_corrects / len(train_loader))
        statisticUtil.addLoss(self.fromNodeId, running_loss / len(train_loader))
        return self.getParam()


if __name__ == '__main__':
    data_root = os.getcwd()
    image_path = data_root + "/DataSet/flower_data/"  # flower data set path
    train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                         transform=Constent.data_transform["train"])

    smallTrain_dataset = Subset(train_dataset, range(0, 100))
    print(len(smallTrain_dataset))
    alexNet = AlexNetTrain(smallTrain_dataset, fromNode=1)
    alexNet.train_main()

    print("\n")

    parameterList = alexNet.getParam()
    print(parameterList)

    for key in parameterList.keys():
        parameterList[key] = parameterList[key] / 2

    print("%%%%%%%%%%%")
    print(parameterList)

    alexNet.updateParam(parameterList)

    print("###########after###########")
    parameterList = alexNet.getParam()
    print(parameterList)
