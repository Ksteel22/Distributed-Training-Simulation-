import mnist as mnist
import numpy as np


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = []
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 3d
        self.last_input_shape = input.shape

        # 3d to 1d
        input = input.flatten()

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        # output before softmax
        # 1d vector
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        # only 1 element of d_L_d_out is nonzero
        for i, gradient in enumerate(d_L_d_out):
            # k != c, gradient = 0
            # k == c, gradient = 1
            # try to find i when k == c
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            # (1000, 1) @ (1, 10) to (1000, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1000, 10) @ (10, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            # it will be used in previous pooling layer
            # reshape into that matrix
            return d_L_d_inputs.reshape(self.last_input_shape)
