from utility import *
import numpy as np

class Layer:

    def __init__(self, in_dim, out_dim, act, mu):
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.random.randn(out_dim)
        self.act = act
        self.mu = mu 

    # DEBUG utility
    def __str__(self):
        return  "matrix:\n" + \
                str(self.w) + \
                "\nbias:\n" + \
                str(self.b) + \
                "\nnode values:\n" + \
                str(self.x)

    def feed_forward(self, x):
        self.x = x
        self.z = np.dot(self.w, self.x) + self.b
        return self.act.function(self.z)

    def propagate_back(self, curr):
        self.cr_step = np.dot(curr, self.act.derivative(self.z))
        return np.dot(self.cr_step, self.w)

    def get_gradient(self):
        ret_w = np.outer(self.cr_step, self.x)
        ret_w += 2 * self.mu * self.w # regularization
        ret_b = self.cr_step
        ret_b += 2 * self.mu * self.b # regularization
        return (ret_w, ret_b)


class Network:
    def __init__(self, size_list, inner_act, last_act, mu):
        self.layers = []
        self.mu = mu
        for i in range (len(size_list) - 2):
            layer = Layer(size_list[i], size_list[i+1], inner_act, mu)
            self.layers.append(layer)

        layer = Layer(size_list[-2], size_list[-1], last_act, mu)
        self.layers.append(layer)

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def propagate_back(self, curr):
        for layer in reversed(self.layers):
            curr = layer.propagate_back(curr)

    def accuracy(self, data):
        right = 0.
        for x, y in data:
            res = self.feed_forward(x)
            choice = np.argmax(res)
            if choice == np.argmax(y):
                right += 1

        return right / len(data)
    
    def mee(self, data):
        ret = 0.
        for x, y in data:
            res = self.feed_forward(x)
            ret += euclideanLoss.function(res, y)

        return ret / len(data)
