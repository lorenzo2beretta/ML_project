from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

val, train, test = read_cup()

lrate = 0.05
epochs = 2000
batch_size = 32
mu = 0.005
beta = 0.9
act_fun = tanh
loss = euclideanLoss

size_list = [10, 40, 2]

now = time.strftime("%c")

network = Network(size_list, act_fun, idn, mu)
losses, val_losses = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size)

'''
PLEASE MAKE THIS BACKWARD-COMPATIBLE WITH PYTHON 2

hyperparams = f"lrate={lrate}\tmu={mu}\tbeta={beta}"
functions = f"activation={act_fun.name}\tloss={loss.name}"
training = f"epochs={epochs}\tbatch_size={batch_size}"

'''

topology = "TOPOLOGIA = " + str(size_list)
train_mee = network.avg_loss(train, loss)
valid_mee = network.avg_loss(val, loss)

'''
with open("experiments_cup.txt", "a") as infile:
    infile.write(now+"\n"+hyperparams+"\n"+functions+"\n"+training+"\n"+topology+f"\ntrain_mee={train_mee}\tvalid_mee={valid_mee}\n\n")

plt.plot(losses, '-', val_losses, 'r--')
plt.show()
'''
