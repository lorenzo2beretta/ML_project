from network import *
import copy
import random

# DELETE THIS F*CKING CLASS, just method train !!

class GradientDescent:
    def __init__(self, loss, lrate, epochs, network):
        self.loss = loss
        self.lrate = lrate
        self.epochs = epochs
        self.network = network

    def get_null_copy(self):
        ret = copy.deepcopy(self.network)
        for layer in ret.layers:
            layer.w.fill(0)
            layer.b.fill(0)
        return ret    

    def train(self, data, validation, beta):
        momentum = self.get_null_copy()
        for epoch in range (self.epochs):
            tmp = self.get_null_copy()            
            loss = 0.
            # acc = 0.
            for x, lb in data:
                y = self.network.feed_forward(x)
                # evaluate loss function differential
                diff = self.loss.derivative(y, lb)
                # if ((y[1] - y[0]) * (lb[1] - lb[0]) > 0):
                #    acc += 1'''
                loss += self.loss.function(y, lb)
                self.network.propagate_back(diff)
                for i, layer in enumerate(self.network.layers):
                    # add gradient to the layer temp data structure
                    grad_w, grad_b = layer.get_gradient()
                    tmp.layers[i].w += grad_w
                    tmp.layers[i].b += grad_b
                    
            for i, layer in enumerate(self.network.layers):
                tmp.layers[i].w /= len(data)
                tmp.layers[i].b /= len(data)
                # updating weights
                momentum.layers[i].w *= beta
                momentum.layers[i].w += (1 - beta) * tmp.layers[i].w
                momentum.layers[i].b *= beta
                momentum.layers[i].b += (1 - beta) * tmp.layers[i].b
                
                layer.w -= self.lrate * momentum.layers[i].w
                layer.b -= self.lrate * momentum.layers[i].b

            loss /= len(data)
            #print("epoch = " + str(epoch) + "\t loss = " + str(loss))

            
            val_loss = 0
            #val_acc = 0.
            for x, lb in validation:
                y = self.network.feed_forward(x)
                val_loss += self.loss.function(y, lb)
                #if ((y[1] - y[0]) * (lb[1] - lb[0]) > 0):
                 #   val_acc += 1

            val_loss /= len(validation)
            #val_acc /= len(validation)
            #acc /= len(data)
            print((epoch, loss, val_loss))
            
    def train_batch(self, data, validation, beta, batch_size):
        momentum = self.get_null_copy()
        for epoch in range (self.epochs):
            random.shuffle(data)
            loss = 0.
            for j in range (len(data) / batch_size):
                tmp = self.get_null_copy()            
                # acc = 0.
                for x, lb in data[j * batch_size: (j + 1) * batch_size]:
                    y = self.network.feed_forward(x)
                    # evaluate loss function differential
                    diff = self.loss.derivative(y, lb)
                    # if ((y[1] - y[0]) * (lb[1] - lb[0]) > 0):
                    #    acc += 1'''
                    loss += self.loss.function(y, lb)
                    self.network.propagate_back(diff)
                    for i, layer in enumerate(self.network.layers):
                        # add gradient to the layer temp data structure
                        grad_w, grad_b = layer.get_gradient()
                        tmp.layers[i].w += grad_w
                        tmp.layers[i].b += grad_b

                for i, layer in enumerate(self.network.layers):
                    tmp.layers[i].w /= batch_size
                    tmp.layers[i].b /= batch_size
                    # updating weights
                    momentum.layers[i].w *= beta
                    momentum.layers[i].w += (1 - beta) * tmp.layers[i].w
                    momentum.layers[i].b *= beta
                    momentum.layers[i].b += (1 - beta) * tmp.layers[i].b

                    layer.w -= self.lrate * momentum.layers[i].w
                    layer.b -= self.lrate * momentum.layers[i].b

            loss /= len(data)
            val_loss = 0
            #val_acc = 0.
            for x, lb in validation:
                y = self.network.feed_forward(x)
                val_loss += self.loss.function(y, lb)
                #if ((y[1] - y[0]) * (lb[1] - lb[0]) > 0):
                 #   val_acc += 1

            val_loss /= len(validation)
            #val_acc /= len(validation)
            #acc /= len(data)
            print((epoch, loss, val_loss))
            
