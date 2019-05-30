from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

'''
This file is devoted to test monks datasets. You can launch a test running  
this script right after substituting the parameters below.

At the end of the training it prints the topology as long as some benchmarks.



------------------------------ PARAMETERS ----------------------------- '''

# use "monks-i" for i = 1, 2, 3
dataset = "monks-3"

# use a custom topolgy
size_list = [17, 15, 1]

# following are self-explained
lrate = 0.01
mu = 0.02
beta = 0.9
epochs = 1500
batch_size = 10

# use one of {tanh, sigmoid, reLU, softMax}
act_fun = tanh

# use one of {binaryCrossEntropy, crossEntropy, squareLoss, euclideanLoss}
loss = binaryCrossEntropy

# use accuracy_multi if you want 2-dim output for classification
acc_fun = accuracy_single

''' ------------------------------------------------------------- '''

network = Network(size_list, act_fun, sigmoid, mu)

val, train, test = read_monks(dataset, single_out=True)

losses, accs, val_losses, val_accs = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size, accuracy=acc_fun)


topology = "TOPOLOGIA = " + str(size_list)
print(topology)
train_acc = network.accuracy(train, acc_fun)
print("train_acc = " + str(train_acc))
valid_acc = network.accuracy(val, acc_fun)
print("valid_acc = " + str(valid_acc))
test_acc = network.accuracy(test, acc_fun)
print("test_acc = " + str(test_acc))
