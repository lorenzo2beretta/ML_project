import numpy as np
import csv

# functions get applied componentwise on their own
class DiffFunction:

    # @param function function to encode
    # @param derivative derivative of the previous function
    def __init__(self, function, derivative, name):
        self.function = function
        self.derivative = derivative
        self.name = name

# -------------- squareLoss function -----------------

def squareLoss_fun(y, lb):
    return np.dot(y - lb, y - lb)

def squareLoss_der(y, lb):
    return 2 * (y - lb)

squareLoss = DiffFunction(squareLoss_fun, squareLoss_der, "MSE")

# ------------ euclideanLoss function --------------

def euclideanLoss_fun(y, z):
    return np.sqrt(np.dot(y - z, y - z))

def euclideanLoss_der(y, z):
    return (y - z) / np.sqrt(np.dot(y - z, y - z))

euclideanLoss = DiffFunction(euclideanLoss_fun, euclideanLoss_der, "MEE")

# -------------- crossEntropy ----------------------

def crossEntropy_fun(y, lb):
    y = np.log(y)
    return - np.dot(y, lb)

def crossEntropy_der(y, lb):
    y = 1. / y
    return - y * lb

crossEntropy = DiffFunction(crossEntropy_fun, crossEntropy_der, "crossEntropy")

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

binCrossEntropy = DiffFunction(binCrossEntropy_fun, binCrossEntropy_der, "binCrossEntropy")

# -------------- sigmoid FUNCTION -----------------

def sigmoid_fun(x):
    sigm = 1./(1.+np.exp(-x))
    return sigm

def sigmoid_der(x):
    sigm = 1./(1.+np.exp(-x))
    return np.diag(sigm*(1.-sigm))

sigmoid = DiffFunction(sigmoid_fun, sigmoid_der, "sigmoid")

# -------------- reLu FUNCTION --------------------

def reLU_fun(x):
    x[x <= 0] = 0
    return x
def reLU_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return np.diag(x)

reLU = DiffFunction(reLU_fun, reLU_der, "reLU")

# ------------- softPlus function -----------------

def softPlus_fun(x):
    return np.log(1 + np.exp(x))

def softPlus_der(x):
    x = np.exp(x)
    return np.diag(x / (1 + x))

softPlus = DiffFunction(softPlus_fun, softPlus_der, "softPlus")

# ---------------- tanh function -------------------

def tanh_fun(x):
    return np.tanh(x)

def tanh_der(x):
    return np.diag(1 - np.tanh(x)**2)

tanh = DiffFunction(tanh_fun, tanh_der, "tanh")

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

softMax = DiffFunction(softMax_fun, softMax_der, "softMax")

# ---------------- Identity -------------------------

def idn_fun(x):
    return x

def idn_der(x):
    return np.identity(x.size)

idn = DiffFunction(idn_fun, idn_der, "id")


# ------------ accuracy metrics ----------
def accuracy_single(y, lb):
    return np.abs(y-lb)<0.5

def accuracy_multi(y, lb):
    return np.argmax(y) == np.argmax(lb)
