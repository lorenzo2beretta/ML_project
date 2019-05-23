from network import *
from gradient_descent import *
from utility import *

import csv
import numpy as np
import random

def generate_random_data():
    data = []
    for i in range (200):
        x = np.random.randint(4, size = 3)
        res = (x[0]==x[1]) and (x[2]==2)
        data.append((x, int(res)))
    return data

def load_monks(filename, val_split=0.05):
    train_data = []
    with open("data/"+filename+".test") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            label = np.array([int(row[1]), 1 - int(row[1])])
            train_data.append( (np.array([int(x) for x in row[2:8]]), label))
    print("Loaded {} datapoints".format(len(train_data)))

    test_data = []
    with open("data/"+filename+".train") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            label = np.array([int(row[1]), 1 - int(row[1])])
            test_data.append((np.array([int(x) for x in row[2:8]]), label) )
    print("Loaded {} test datapoints".format(len(test_data)))

    n = int(val_split*len(train_data))
    random.shuffle(train_data)

    return train_data[:n], train_data[n:], test_data


val, train, test = load_monks("monks-1")

lrate = 0.01
mu = 0.000
epochs = 1000
beta = 0.95
size_list = [6, 10, 2]
network = Network(size_list, reLU, softMax, mu)

algo = GradientDescent(crossEntropy, lrate, epochs, network)
algo.train(train, val, beta)

print("Accuracy validation = " + str(network.accuracy(val)))
print("Test validation = " + str(network.accuracy(test)))

print(network.feed_forward(np.array([1, 1, 0, 0, 0, 0])))
print(network.feed_forward(np.array([1, 3, 2, 0, 1, 1])))
print(network.feed_forward(np.array([3, 0, 2, 1, 4, 1])))

# print(network.feed_forward(np.array([0, 1])))
# print(network.feed_forward(np.array([1, 0])))
# print(network.feed_forward(np.array([0, 0])))
# print(network.feed_forward(np.array([1, 1])))
