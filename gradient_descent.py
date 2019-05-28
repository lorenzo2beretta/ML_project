from network import *
import random
import math

''' -------- gradient_descent ---------

This function implements batched gradient descent with momentum.
Returns a tuple of lists containing loss and accuracy statistics on training  
and validation set. 

Keyword Arguments:

train -- a numpy array of couples (x, y) constituting the training set

val -- a numpy array of couples (x, y) constituting the validation set

beta -- momentum exponential reduction constant

loss -- an object of DiffFunction class constituting the loss function

lrate -- learning rate (usually denoted with eta)

epochs -- number of complete train set parsing cycles

net -- Network object to train

bsize -- size of a single batch

accuracy -- a function defining the outcome of a classification task

'''
def gradient_descent(train, val, beta, loss, lrate, epochs, net, bsize = None, accuracy = None):
    # single-batched case
    if not bsize:
        bsize = len(train)

    # get a copy of the network to store momentum   
    momentum_net = net.get_null_copy()
    
    # list of data to return and eventually plot later
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # cycling through epochs
    for epoch in range (epochs):
        random.shuffle(train)
        
        # cycling through batches
        for j in range (int(math.ceil(len(train) / bsize))):
            grad_net = net.get_null_copy()

            # cycling within a batch
            for x, lb in train[j * bsize: (j + 1) * bsize]:
                y = net.feed_forward(x)
                diff = loss.derivative(y, lb)
                net.propagate_back(diff)
                
                # updating gradients
                for i, layer in enumerate(net.layers):
                    grad_w, grad_b = layer.get_gradient()
                    grad_net.layers[i].w += grad_w
                    grad_net.layers[i].b += grad_b

            for i, layer in enumerate(net.layers):
                grad_net.layers[i].w /= bsize
                grad_net.layers[i].b /= bsize
                # momentum update
                momentum_net.layers[i].w *= beta
                momentum_net.layers[i].w += (1 - beta) * grad_net.layers[i].w
                momentum_net.layers[i].b *= beta
                momentum_net.layers[i].b += (1 - beta) * grad_net.layers[i].b
                # weights update
                layer.w -= lrate * momentum_net.layers[i].w
                layer.b -= lrate * momentum_net.layers[i].b

        # loss  evaluation
        train_loss = net.avg_loss(train, loss)
        val_loss = net.avg_loss(val, loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if accuracy:
            # accuracy evaluation
            train_acc = net.accuracy(train, accuracy)
            val_acc = net.accuracy(val, accuracy)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            # print statistics 
            print((epoch, train_loss, train_acc, val_loss, val_acc))
        else:
            print((epoch, train_loss, val_loss))

    # return statistics to plot  
    if accuracy:
        return (train_losses, train_accuracies, val_losses, val_accuracies)
    else:
        return (train_losses, val_losses)
