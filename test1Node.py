import mnist

import Constent
from CnnTrain import CnnTrain
from StatisticUtil import statisticUtil

data = {"images": mnist.train_images()[:Constent.DATA_NUM], "labels": mnist.train_labels()[:Constent.DATA_NUM]}
cnnTrain = CnnTrain(Constent.NUM_FILTERS, Constent.INPUT_LEN, Constent.NODES, data, 0)

paramList = {"conv_filters": Constent.CONV_FILTERS,
             "softmax_weights": Constent.SOFTMAX_WEIGHTS,
             "softmax_biases": Constent.SOFTMAX_BIASES}
cnnTrain.updateParam(paramList)
statisticUtil.initialize(1)

for i in range(Constent.TRAIN_TIMES):
    print("------EPOCH {} -----".format(i))
    cnnTrain.train_main()


cnnTrain.test_cnn()
statisticUtil.output2Csv()

