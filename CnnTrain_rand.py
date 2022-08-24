from random import random

from CnnTrain import CnnTrain
from StatisticUtil import statisticUtil


class CnnTrain_rand(CnnTrain):
    def __init__(self, num_filters, in_length, out_nodes, data, fromNode, nodeNum):
        super(CnnTrain_rand, self).__init__(num_filters, in_length, out_nodes, data, fromNode)
        self.nodeNum = nodeNum

    def train_main(self):
        train_images_temp = []
        train_labels_temp = []

        for i in range(len(self.train_images)):
            if random() < 1 / self.nodeNum:
                train_images_temp.append(self.train_images[i])
                train_labels_temp.append(self.train_labels[i])

        # Train
        loss = 0
        num_correct = 0
        # i: index
        # im: image
        # label: label
        for i, (im, label) in enumerate(zip(train_images_temp, train_labels_temp)):
            # if i > 0 and i % 100 == 99:
            #     print(
            #         '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            #         (i + 1, loss / 100, num_correct)
            #     )
            #     loss = 0
            #     num_correct = 0

            l, acc = self.train(im, label)
            loss += l
            num_correct += acc

        statisticUtil.addAcc(self.fromNodeId, num_correct / len(train_images_temp))
        statisticUtil.addLoss(self.fromNodeId, loss / len(train_images_temp))
        return self.getParam()
