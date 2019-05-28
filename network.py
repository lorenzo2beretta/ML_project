from utility import *
import copy
import numpy as np

''' ------------------------- Layer -----------------------------
This class consists of a fully connected layer to be used in a NN.
It implements three basic methods for feeding the network, apply the  
back-propagation algorithm and evaluate the gradient.

Fields:

w -- matrix of weights
b -- vector of biases
act -- activation function associated to forward neurons
mu -- regularization parameter

'''
class Layer:

    def __init__(self, in_dim, out_dim, act, mu):
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.random.randn(out_dim)
        self.act = act
        self.mu = mu

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

    
''' ------------------------- Network ---------------------------
This class consist of a NN implementing feed-forward and back-propagation.
It has also attached some utility methods to get a null-initialized copy  
of its shape and to benchmark its performance over a labelled dataset.

Fields:

layers -- a list of Layer class objects

mu -- regularization parameter

'''
class Network:

    ''' Initialize the Network discriminating inner layer activation 
        functions form the function employed in the last one, since
        it may involve some ad hoc squashing function.
    '''
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

    ''' Returns an isomorphic copy which layers' fields are zero initialized  
    '''
    def get_null_copy(self):
        ret = copy.deepcopy(self)
        for layer in ret.layers:
            layer.w.fill(0)
            layer.b.fill(0)
        return ret

    ''' A couple of function to benchmark Network's performance
    '''
    def accuracy(self, data, accuracy):
        right = 0.
        for x, y in data:
            res = self.feed_forward(x)
            if accuracy(res, y):
                right += 1

        return right / len(data)

    def avg_loss(self, data, loss):
        ret = 0.
        for x, y in data:
            res = self.feed_forward(x)
            ret += loss.function(res, y)

        return ret / len(data)
