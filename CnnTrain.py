import mnist
import numpy as np

import Constent
from StatisticUtil import statisticUtil
from TrainModel import TrainModel
from cnn.Conv3x3 import Conv3x3
from cnn.MaxPool2 import MaxPool2
from cnn.Softmax import Softmax


class CnnTrain(TrainModel):
    def __init__(self, num_filters, in_length, out_nodes, data, fromNode):
        # self.test_images = mnist.test_images()[:1000]
        # self.test_labels = mnist.test_labels()[:1000]

        # We only use the first 1k examples of each set in the interest of time.
        # Feel free to change this if you want.
        # self.train_images = mnist.train_images()[:1000]
        # self.train_labels = mnist.train_labels()[:1000]
        self.fromNodeId = fromNode

        self.train_images = data["images"]
        self.train_labels = data["labels"]
        self.test_images = Constent.test_images
        self.test_labels = Constent.test_labels

        self.conv = Conv3x3(num_filters)  # 28x28x1 -> 26x26x8
        self.pool = MaxPool2()  # 26x26x8 -> 13x13x8
        self.softmax = Softmax(in_length, out_nodes)  # 13x13x8 -> 10

    def updateParam(self, paramList):
        """
        update train parameter with input paramList
        :param paramList:
        :return:
        """
        self.conv.filters = paramList["conv_filters"]
        self.softmax.weights = paramList["softmax_weights"]
        self.softmax.biases = paramList["softmax_biases"]
        pass

    def getParam(self):
        """
        output current parameter list
        :return: paramList:conv_filters, softmax_weights, softmax_biases
        """
        paramList = {"conv_filters": self.conv.filters, "softmax_weights": self.softmax.weights,
                     "softmax_biases": self.softmax.biases}
        return paramList

    def forward(self, image, label):
        """
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        """
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

        # out: vector of probability
        # loss: num
        # acc: 1 or 0

    def train(self, im, label, lr=0.01):
        """
        train a image
        :param im: input image
        :param label: this image's label
        :param lr: learning rate default is 0.01
        :return:
        """
        # Forward
        out, loss, acc = self.forward(im, label)

        # Calculate intial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # print("gradient[" + str(label) + "]:" + str(gradient[label]))

        # Backprop
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, lr)

        return loss, acc

    def train_main(self):
        """
        start train all data
        and return paramList after train
        :return: paramList after train
        """
        # Shuffle the training data
        permutation = np.random.permutation(len(self.train_images))
        train_images = self.train_images[permutation]
        train_labels = self.train_labels[permutation]

        # Train
        loss = 0
        num_correct = 0
        # i: index
        # im: image
        # label: label
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
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
        print("acc: " + str(num_correct / len(train_images)))
        print("loss: " + str(loss / len(train_images)))
        # statisticUtil.addAcc(self.fromNodeId, num_correct / len(train_images))
        # statisticUtil.addLoss(self.fromNodeId, loss / len(train_images))
        self.test_cnn()

        return self.getParam()

    def test_cnn(self):
        """
        test current parameters using default test data
        :return:
        """
        # Test the CNN
        print('\n--- Testing the CNN ---')
        loss = 0
        num_correct = 0
        outList = []
        for im, label in zip(self.test_images, self.test_labels):
            out, l, acc = self.forward(im, label)
            loss += l
            num_correct += acc
            outList.append(out)

        num_tests = len(self.test_images)

        print('Test Loss:', loss / num_tests)
        print('Test Accuracy:', num_correct / num_tests)
        # print("out: " + str(outList))
        statisticUtil.addAcc(self.fromNodeId, num_correct / len(self.test_images))
        statisticUtil.addLoss(self.fromNodeId, loss / len(self.test_images))

        return loss / num_tests, num_correct / num_tests
