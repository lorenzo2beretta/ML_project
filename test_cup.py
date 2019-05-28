from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def preprocess_cup(val_split=0.25):
    train_data = []
    with open("cup/cup.train") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            label = np.array([float(row[10]), float(row[11])])
            data = np.array([float(x) for x in row[0:10]])
            train_data.append((data, label))
    print("Loaded {} train datapoints".format(len(train_data)))

    test_data = []
    with open("cup/cup.test") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            label = np.array([float(row[10]), float(row[11])])
            data = np.array([float(x) for x in row[0:10]])
            test_data.append((data, label))
    print("Loaded {} test datapoints".format(len(test_data)))

    n = int(val_split*len(train_data))
    random.shuffle(train_data)

    return train_data[:n], train_data[n:], test_data

val, train, test = preprocess_cup()

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

algo = GradientDescent(loss, lrate, epochs, network)
losses, val_losses = algo.train_batch(train, val, beta, batch_size)

hyperparams = f"lrate={lrate}\tmu={mu}\tbeta={beta}"
functions = f"activation={act_fun.name}\tloss={loss.name}"
training = f"epochs={epochs}\tbatch_size={batch_size}"

topology = "TOPOLOGIA = " + str(size_list)
train_mee = network.mee(train)
valid_mee = network.mee(val)

with open("experiments_cup.txt", "a") as infile:
    infile.write(now+"\n"+hyperparams+"\n"+functions+"\n"+training+"\n"+topology+f"\ntrain_mee={train_mee}\tvalid_mee={valid_mee}\n\n")

plt.plot(losses, '-', val_losses, 'r--')
plt.show()
