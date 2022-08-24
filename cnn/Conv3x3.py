import numpy as np


class Conv3x3:
    # 卷积层使用3*3的filter.
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = []

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):  # (h-2)/(w-2)是滤波以单位为1的步长，所需要移动的步数
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]  # （i+3） 3*3的filter所移动的区域
                yield im_region, i, j

    def forward(self, input):
        # 28x28
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))  # 创建一个（h-2）*（w-2）的零矩阵用于填充每次滤波后的值

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output  # 4*4的矩阵经过3*3的filter后得到一个2*2的矩阵

    def backprop(self, d_L_d_out, learn_rate):
        # d_L_d_out: the loss gradient for this layer's outputs
        # learn_rate: a float
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None
