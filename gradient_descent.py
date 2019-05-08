from network import *

# DELETE THIS F*CKING CLASS, hust method train !!

class GradientDescent:
    def __init__(self, loss, lrate, epochs, network):
        self.loss = loss
        self.lrate = lrate
        self.epochs = epochs
        self.network = network

    # @param data is a list of pairs (input_list, output_list)
    def train(self, data):
        for i in range (self.epochs):
            # reset all weights
            self.network.reset_all()
            loss = 0
            for x, lb in data:
                y = self.network.feed_forward(x)
                # evaluate loss function differential
                diff = self.loss.derivative(y, lb)
                loss += self.loss.function(y, lb)
                self.network.propagate_back(diff)
                for layer in self.network.layers:
                    # add gradient to the layer temp data structure
                    grad_w, grad_b = layer.get_gradient()
                    layer.tmp_w += grad_w
                    layer.tmp_b += grad_b

            for layer in self.network.layers:
                layer.tmp_w /= len(data)
                layer.tmp_b /= len(data)
                # updating weights
                layer.w -= self.lrate * layer.tmp_w
                layer.b -= self.lrate * layer.tmp_b

            loss /= len(data)
            print((i, loss))
