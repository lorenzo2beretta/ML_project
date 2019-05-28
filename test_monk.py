from network import *
from gradient_descent import *
from utility import *

import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

val, train, test = read_monks("monks-3")

lrate = 0.01
mu = 0.005
beta = 0.9
epochs = 5000
batch_size = 32

act_fun = tanh

dataset = "monks-3"
single = True

if single:
    size_list = [17, 15, 1]
    network = Network(size_list, act_fun, sigmoid, mu)
    loss = binCrossEntropy
    acc_fun = accuracy_single
else:
    size_list = [17, 15, 2]
    network = Network(size_list, act_fun, softMax, mu)
    loss = crossEntropy
    acc_fun = accuracy_multi

now = time.strftime("%c")

val, train, test = read_monks(dataset, single_out=single, val_split=0.1)


losses, accs, val_losses, val_accs = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size, accuracy=acc_fun)

'''
PLEASE MAKE THIS BACKWARD-COMPATIBLE WITH PYTHON 2

hyperparams = f"lrate={lrate}\tmu={mu}\tbeta={beta}"
functions = f"activation={act_fun.name}\tloss={loss.name}"
training = f"epochs={epochs}\tbatch_size={batch_size}"

'''

topology = "TOPOLOGIA = " + str(size_list)
train_acc = network.accuracy(train, acc_fun)
valid_acc = network.accuracy(val, acc_fun)
test_acc = network.accuracy(test, acc_fun)
print(test_acc)

'''

with open("experiments_monks.txt", "a") as infile:
    infile.write(f"{dataset} @ {now}"+"\n" + hyperparams+"\n"+functions+"\n"+training+"\n"+topology + f"\ntrain_acc={train_acc}\tvalid_acc={valid_acc}\ttest_acc={test_acc}\n\n")

plt.subplot(211)
plt.plot(losses, '-', val_losses, 'r--')
plt.subplot(212)
plt.plot(accs, '-', val_accs, 'r--')
plt.savefig(f"monks/plots/{dataset}_{now}.png")
plt.show()

'''
