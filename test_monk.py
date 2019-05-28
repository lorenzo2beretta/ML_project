from network import *
from gradient_descent import *
from utility import *

import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def preprocess_monks(filename, val_split=0.05, single_out=False):
    train_data = []
    with open("monks/"+filename+".train") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            if single_out:
                label = np.array([int(row[1])])
            else:
                label = np.array([int(row[1]), 1 - int(row[1])])
            data = np.zeros(17)
            data[int(row[2]) - 1] = 1
            data[int(row[3]) + 2] = 1
            data[int(row[4]) + 5] = 1
            data[int(row[5]) + 7] = 1
            data[int(row[6]) + 10] = 1
            data[int(row[7]) + 14] = 1
            train_data.append((data, label))
    print("Loaded {} train datapoints".format(len(train_data)))

    test_data = []
    with open("monks/"+filename+".test") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            if single_out:
                label = np.array([int(row[1])])
            else:
                label = np.array([int(row[1]), 1 - int(row[1])])
            data = np.zeros(17)
            data[int(row[2]) - 1] = 1
            data[int(row[3]) + 2] = 1
            data[int(row[4]) + 5] = 1
            data[int(row[5]) + 7] = 1
            data[int(row[6]) + 10] = 1
            data[int(row[7]) + 14] = 1
            test_data.append((data, label))
    print("Loaded {} test datapoints".format(len(test_data)))

    n = int(val_split*len(train_data))
    random.shuffle(train_data)

    return train_data[:n], train_data[n:], test_data


lrate = 0.008
mu = 0.01
beta = 0.9

epochs = 5000
batch_size = 20

act_fun = tanh

single = False

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

val, train, test = preprocess_monks("monks-3", single_out=single, val_split=0.1)

algo = GradientDescent(loss, lrate, epochs, network)

losses, accs, val_losses, val_accs = algo.train_batch(train, val, beta, batch_size, accuracy=acc_fun)

hyperparams = f"lrate={lrate}\tmu={mu}\tbeta={beta}"
functions = f"activation={act_fun.name}\tloss={loss.name}"
training = f"epochs={epochs}\tbatch_size={batch_size}"

topology = "TOPOLOGIA = " + str(size_list)
train_acc = network.accuracy(train, acc_fun)
valid_acc = network.accuracy(val, acc_fun)
test_acc = network.accuracy(test, acc_fun)
print(test_acc)



with open("experiments_monks.txt", "a") as infile:
    infile.write(now+"\n"+hyperparams+"\n"+functions+"\n"+training+"\n"+topology+f"\ntrain_acc={train_acc}\tvalid_acc={valid_acc}\ttest_acc={test_acc}\n\n")

plt.subplot(211)
plt.plot(losses, '-', val_losses, 'r--')
plt.subplot(212)
plt.plot(accs, '-', val_accs, 'r--')
plt.show()
