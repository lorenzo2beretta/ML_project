from network import *
from gradient_descent import *
from utility import *

'''
This file is devoted to test CUP dataset. You can launch a test running  
this script right after substituting the parameters below.

At the end of the training it prints the topology as long as some benchmarks.

'''

''' --------------------- PARAMETERS ----------------------------- '''

# use a custom topolgy
size_list = [10, 20, 2]

# following are self-explained
lrate = 0.01
mu = 0.001
beta = 0
epochs = 2000
batch_size = 32

# use one of {tanh, sigmoid, reLU, softMax}
act_fun = sigmoid

# use one of {binaryCrossEntropy, crossEntropy, squareLoss, euclideanLoss}
loss = euclideanLoss

''' ------------------------------------------------------------- '''

network = Network(size_list, act_fun, sigmoid, mu)

val, train, test = read_cup()

losses, val_losses = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size)


topology = "TOPOLOGIA = " + str(size_list)
print(topology)
train_loss = network.avg_loss(train, loss)
print("train_loss = " + str(train_loss))
valid_loss = network.avg_loss(val, loss)
print("valid_loss = " + str(valid_loss))
test_loss = network.avg_loss(test, loss)
print("test_loss = " + str(test_loss))
