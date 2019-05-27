from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random

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

lrate = 0.1
epochs = 2000
mu = 0.001
beta = 0.95

size_list = [10, 40, 2]
network = Network(size_list, tanh, idn, mu)

algo = GradientDescent(euclideanLoss, lrate, epochs, network)
algo.train_batch(train, val, beta, 30)

print("TOPOLOGIA = " + str(size_list))
print("Train = " + str(network.mee(train))) 
print("Validation = " + str(network.mee(val)))

# Evaluating average norm of outputs
norm = 0.
for x, y in train:
    norm += euclideanLoss.function(y, np.zeros(2))

norm /= len(train)

print("Avg Norm = " + str(norm))
