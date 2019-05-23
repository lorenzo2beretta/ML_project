import numpy as np
import csv

# functions get applied componentwise on their own
class DiffFunction:

    # @param function function to encode
    # @param derivative derivative of the previous function
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

# -------------- SQUARE LOSS FUNCTION --------------

def squareLoss_fun(y, lb):
    return np.dot(y - lb, y - lb)

def squareLoss_der(y, lb):
    return 2 * (y - lb)

squareLoss = DiffFunction(squareLoss_fun, squareLoss_der)

# -------------- crossEntropy ----------------------

def crossEntropy_fun(y, lb):
    y = np.log(y)
    return - np.dot(y, lb)

def crossEntropy_der(y, lb):
    y = 1. / y
    return - y * lb

crossEntropy = DiffFunction(crossEntropy_fun, crossEntropy_der)

# -------------- binaryCrossEntropy ----------------------

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
        
binCrossEntropy = DiffFunction(binCrossEntropy_fun, binCrossEntropy_der)

# -------------- sigmoid FUNCTION -----------------

def sigmoid_fun(x):
    sigm = 1./(1.+np.exp(-x))
    return sigm

def sigmoid_der(x):
    sigm = 1./(1.+np.exp(-x))
    return np.diag(sigm*(1.-sigm))

sigmoid = DiffFunction(sigmoid_fun, sigmoid_der)

# -------------- reLu FUNCTION --------------------

def reLU_fun(x):
    x[x <= 0] = 0
    return x
def reLU_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return np.diag(x)

reLU = DiffFunction(reLU_fun, reLU_der)

# -------------- softMax FUNCTION ------------------

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

softMax = DiffFunction(softMax_fun, softMax_der)

    # ------------- function to read very data  ---------------

def read_monks(file_path):
    data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            row = map(int, row[1:8])
            data.append((np.array(row[1:7]), np.array([row[0],1 - row[0]])))
    return data