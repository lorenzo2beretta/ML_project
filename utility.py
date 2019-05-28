import numpy as np
import csv

'''
This file contains some utility functions. Here is defined the DiffFunction  
class, consisting of a function and its derivative. Then many DiffFunction  
objects are defined: those are activation and loss functions.

Then we can find two utility function to assess the accuracy of a NN  
classifier. 

Finally some more utilty function to read input are included.


'''

''' ----------- DiffFunction ------------
This class represents a piecewise differentiable function and its derivative.

After its definition a long list loss and activation functions as  
DiffFunction objects.

'''
class DiffFunction:
    def __init__(self, function, derivative, name):
        self.function = function
        self.derivative = derivative
        self.name = name

''' ------------------------------------------------------------
 ----------------------- LOSS FUNCTIONS ------------------------
---------------------------------------------------------------- '''
        
''' -------------- squareLoss ----------------- '''

def squareLoss_fun(y, lb):
    return np.dot(y - lb, y - lb)

def squareLoss_der(y, lb):
    return 2 * (y - lb)

squareLoss = DiffFunction(squareLoss_fun, squareLoss_der, "MSE")

''' ------------ euclideanLoss -------------- '''

def euclideanLoss_fun(y, z):
    return np.sqrt(np.dot(y - z, y - z))

def euclideanLoss_der(y, z):
    return (y - z) / np.sqrt(np.dot(y - z, y - z))

euclideanLoss = DiffFunction(euclideanLoss_fun, euclideanLoss_der, "MEE")

''' -------------- crossEntropy ------------------- '''

def crossEntropy_fun(y, lb):
    y = np.log(y)
    return - np.dot(y, lb)

def crossEntropy_der(y, lb):
    y = 1. / y
    return - y * lb

crossEntropy = DiffFunction(crossEntropy_fun, crossEntropy_der, "crossEntropy")

''' -------------- binaryCrossEntropy -------------- '''

def binCrossEntropy_fun(y, lb):
    if lb == 0:
        return - np.log(1. - y)
    else:
        return - np.log(y)

def binCrossEntropy_der(y, lb):
    if lb == 0:
        return 1 / (1. - y)
    else:
        return - 1. / y

binCrossEntropy = DiffFunction(binCrossEntropy_fun, binCrossEntropy_der, "binCrossEntropy")

''' ------------------------------------------------------------
 ----------------------- ACTIVATION FUNCTIONS ------------------
---------------------------------------------------------------- '''


''' -------------- sigmoid ----------------- '''

def sigmoid_fun(x):
    sigm = 1./(1.+np.exp(-x))
    return sigm

def sigmoid_der(x):
    sigm = 1./(1.+np.exp(-x))
    return np.diag(sigm*(1.-sigm))

sigmoid = DiffFunction(sigmoid_fun, sigmoid_der, "sigmoid")

''' -------------- reLu -------------------- '''

def reLU_fun(x):
    x[x <= 0] = 0
    return x
def reLU_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return np.diag(x)

reLU = DiffFunction(reLU_fun, reLU_der, "reLU")

''' ------------- softPlus ----------------- '''

def softPlus_fun(x):
    return np.log(1 + np.exp(x))

def softPlus_der(x):
    x = np.exp(x)
    return np.diag(x / (1 + x))

softPlus = DiffFunction(softPlus_fun, softPlus_der, "softPlus")

''' ---------------- tanh -------------------- '''

def tanh_fun(x):
    return np.tanh(x)

def tanh_der(x):
    return np.diag(1 - np.tanh(x)**2)

tanh = DiffFunction(tanh_fun, tanh_der, "tanh")

''' -------------- softMax ------------------ '''

def softMax_fun(x):
    x = np.exp(x)
    x /= np.sum(x)
    return x

def softMax_der(x):
    ret = np.zeros((len(x), len(x)))
    x = np.exp(x)
    s = np.sum(x)
    for i in range (len(x)):
        for j in range (len(x)):
            ret[i][j] = - x[i] * x[j] / (s ** 2)
            if i == j:
                ret[i][j] += x[j] / s

    return ret

softMax = DiffFunction(softMax_fun, softMax_der, "softMax")

''' ---------------- Identity ------------------ '''

def idn_fun(x):
    return x

def idn_der(x):
    return np.identity(x.size)

idn = DiffFunction(idn_fun, idn_der, "id")

'''-------------------- ACCURACY FUNCTIONS ------------------------

This couple of functions provides a 0-1 valued assessment of the
output of a classification network.

'''
def accuracy_single(y, lb):
    if np.abs(y-lb)<0.5:
        return 1
    else:
        return 0

def accuracy_multi(y, lb):
    if np.argmax(y) == np.argmax(lb):
        return 1
    else:
        return 0

''' ---------------------- READ UTILITIES -------------------- '''
    
'''--------- read_monks ---------
This function read monks-i.test and monks-i.train for i = 1, 2, 3.
Moreover it encodes one-hot the discrete values of input.

Keyword Arguments:

filename -- stirng containing "monks-i" for i = 1, 2, 3

val_split -- fraction of training devoted to validation

single_out -- boolean to decide how to encode output

'''
def read_monks(filename, val_split=0.05, single_out=False):
    # Reading train data
    train_data = []
    with open("monks/"+filename+".train") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:

            if single_out:
                label = np.array([int(row[1])])
            else:
                label = np.array([int(row[1]), 1 - int(row[1])])

            # One-Hot encoding
            data = np.zeros(17)
            data[int(row[2]) - 1] = 1
            data[int(row[3]) + 2] = 1
            data[int(row[4]) + 5] = 1
            data[int(row[5]) + 7] = 1
            data[int(row[6]) + 10] = 1
            data[int(row[7]) + 14] = 1
            train_data.append((data, label))
    print("Loaded {} train datapoints".format(len(train_data)))

    # Reading test data
    test_data = []
    with open("monks/"+filename+".test") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:

            if single_out:
                label = np.array([int(row[1])])
            else:
                label = np.array([int(row[1]), 1 - int(row[1])])

            # One-Hot encoding
            data = np.zeros(17)
            data[int(row[2]) - 1] = 1
            data[int(row[3]) + 2] = 1
            data[int(row[4]) + 5] = 1
            data[int(row[5]) + 7] = 1
            data[int(row[6]) + 10] = 1
            data[int(row[7]) + 14] = 1
            test_data.append((data, label))
            
    # Splitting training data between train and validation randomly 
    n = int(val_split*len(train_data))
    random.shuffle(train_data)
    return train_data[:n], train_data[n:], test_data

    
'''--------- read_cup ---------
This function read cup.train and cup.test. We produced those files from the  
original training set since we needed an internal test set.

Keyword Arguments:

val_split -- fraction of training devoted to validation

'''
def read_cup(val_split=0.25):
    # Reading train data
    train_data = []
    with open("cup/cup.train") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            label = np.array([float(row[10]), float(row[11])])
            data = np.array([float(x) for x in row[0:10]])
            train_data.append((data, label))

    # Reading test data
    test_data = []
    with open("cup/cup.test") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            label = np.array([float(row[10]), float(row[11])])
            data = np.array([float(x) for x in row[0:10]])
            test_data.append((data, label))

    # Splitting training data between train and validation randomly 
    n = int(val_split*len(train_data))
    random.shuffle(train_data)
    return train_data[:n], train_data[n:], test_data
