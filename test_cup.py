from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
#import matplotlib.pyplot as plt


def run_one_train(size_list, act_fun, loss, eval_loss, lrate, mu, beta, epochs, batch_size, debug=False):
    val, train, test = read_cup()
    now = time.strftime("%c")

    network = Network(size_list, act_fun, idn, mu)

    losses, val_losses = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size, debug=debug)

    '''
    hyperparams = f"lrate={lrate}\tmu={mu}\tbeta={beta}"
    functions = f"activation={act_fun.name}\tloss={loss.name}"
    training = f"epochs={epochs}\tbatch_size={batch_size}"

    topology = "TOPOLOGIA = " + str(size_list)
    train_mee = network.avg_loss(train, loss)
    valid_mee = network.avg_loss(val, loss)

    with open("experiments_cup.txt", "a") as infile:
        infile.write(f"{now}"+"\n" + hyperparams+"\n"+functions+"\n"+training+"\n"+topology + f"\ntrain_mee={train_mee}\tvalid_mee={valid_mee}\n\n")


    plt.figure()
    plt.plot(losses, '-', val_losses, 'r--')
    plt.legend(["loss","val_loss"])
    plt.title("Loss")

    plt.savefig(f"cup/plots/{now}.png", dpi=300)
    if debug:
        plt.show()
    '''

    valid_mee = network.avg_loss(val, eval_loss)
    return valid_mee


def grid_search(act_fun, loss, epochs=1500, batch_size=32):
    results = []
    size_list = [10, 20, 2]
    for lrate in [0.1, 0.05, 0.01, 0.005, 0.001]:
        for mu in [0.005, 0.001, 0.0005, 0.0001, 0.00005]:
            for beta in [0, 0.8, 0.95]:
                res = run_one_train(size_list, act_fun, loss, euclideanLoss, lrate, mu, beta, epochs, batch_size, debug=False)
                print(res, lrate, mu, beta)
                results.append((res, (lrate, mu, beta)))
    results.sort()
    print(results[:10])



epochs = 1000
batch_size = 32

lrate = 0.05
mu = 0.005
beta = 0.9
act_fun = tanh
loss = euclideanLoss

size_list = [10, 10, 2]

#grid_search(hidden, act_fun, loss, epochs, batch_size)
