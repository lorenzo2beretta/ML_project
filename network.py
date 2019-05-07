import numpy as np


# functions get applied componentwise on their own
class DiffFunction:

    # @param function function to encode
    # @param derivative derivative of the previous function
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative



class Layer:

    # @param in_dim number of inward nodes
    # @param out_dim number of outward nodes
    # @param act an ActFun object encoding the activation function
    def __init__(self, in_dim, out_dim, act):
        self.w = np.random.rand(out_dim, in_dim)
        self.b = np.random.rand(out_dim)
        self.act = act
        self.tmp_w = np.zeros((out_dim, in_dim))
        self.tmp_b = np.zeros((out_dim))

    def reset(self):
        self.tmp_w.fill(0)
        self.tmp_b.fill(0)

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
        self.cr_step = curr * self.act.derivative(self.z)
        return np.dot(self.cr_step, self.w)

    def get_gradient(self):
        ret_w = np.outer(self.cr_step, self.x)
        ret_b = self.cr_step
        return (ret_w, ret_b)


class Network:
    def __init__(self, size_list, inner_act, last_act):
        self.layers = []
        for i in range (len(size_list) - 2):
            layer = Layer(size_list[i], size_list[i+1], inner_act)
            self.layers.append(layer)

        layer = Layer(size_list[-2], size_list[-1], last_act)
        self.layers.append(layer)

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x

    def propagate_back(self, curr):
        for layer in reversed(self.layers):
            curr = layer.propagate_back(curr)

    def reset_all(self):
        for layer in self.layers:
            layer.reset()
