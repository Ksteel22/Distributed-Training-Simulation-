import Constent
from CnnTrain_rand import CnnTrain_rand
from WorkNode import WorkNode


class WorkNode_WithSameData_rand(WorkNode):
    def __init__(self, nodeId, neighborList, bandwidth, fixedDelay, data):
        self.nodeId = nodeId
        self.neighborList = neighborList
        self.bandwidth = bandwidth
        self.fixedDelay = fixedDelay
        # self.data = data
        self.simuTime = 0
        self.tempParamList = {}
        self.lastReceiveTime = 0

        input_len = 13 * 13 * 8
        nodes = 10
        num_filters = 8
        self.cnnTrain = CnnTrain_rand(num_filters, input_len, nodes, data, self.nodeId, Constent.NODE_NUM)
        self.paramList = {"conv_filters": Constent.CONV_FILTERS,
                          "softmax_weights": Constent.SOFTMAX_WEIGHTS,
                          "softmax_biases": Constent.SOFTMAX_BIASES}



