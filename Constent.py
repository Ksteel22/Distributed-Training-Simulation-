# import mnist
import numpy as np
from torchvision import transforms, datasets, utils
TRAIN_SAMPLE_SIZE = 50
TRAIN_TIMES = 100
# NODE_NUM = 10
DATA_NUM = 1000

INPUT_LEN = 13 * 13 * 8
NODES = 10
NUM_FILTERS = 8
CONV_FILTERS = np.random.randn(8, 3, 3) / 9
SOFTMAX_WEIGHTS = np.random.randn(INPUT_LEN, NODES) / INPUT_LEN
SOFTMAX_BIASES = np.zeros(NODES)
#
# test_images = mnist.test_images()[:100]
# test_labels = mnist.test_labels()[:100]
HAS_PARAMETER = False
PARAMETER = {}

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
