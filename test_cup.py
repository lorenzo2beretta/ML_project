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
mu = 0.001
epochs = 3000
beta = 0.95

size_list = [10, 15, 2]
network = Network(size_list, sigmoid, idn, mu)

algo = GradientDescent(squareLoss, lrate, epochs, network)
algo.train(train, val, beta)
print("TOPOLOGIA = " + str(size_list))
print("Train = " + str(network.accuracy(train))) 
print("Validation = " + str(network.accuracy(val)))
print("Test = " + str(network.accuracy(test)))



    
                            
