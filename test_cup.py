from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pickle


def run_one_train(size_list, act_fun, loss, lrate, mu, beta, epochs, batch_size, debug=False):
    val, train, test = read_cup()
    now = time.strftime("%c")

    network = Network(size_list, act_fun, idn, mu)

    losses, val_losses = gradient_descent(train, val, beta, loss, lrate, epochs, network, batch_size, debug=debug)

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

    valid_mee = network.avg_loss(val, loss)
    if valid_mee < 1.20:
        with open(f"models/model_{valid_mee}.pkl", "wb") as f:
            pickle.dump(network, f)

    return (valid_mee, network)


def grid_search(act_fun, beta, epochs=1500, batch_size=32):
    results = []
    size_list = [10, 20, 2]
    for lrate in [0.1, 0.05, 0.01, 0.005, 0.001]:
        for mu in [0.005, 0.001, 0.0005, 0.0001]:
                tot = 0.
                for _ in range(10):
                    tot += run_one_train(size_list, act_fun, euclideanLoss, lrate, mu, beta, epochs, batch_size, debug=False)
                tot /= 10
                print(tot, lrate, mu, beta)
                results.append((tot, (lrate, mu, beta)))
    results.sort()
    with open("test_cup_{}_{}.txt".format(act_fun.name, str(beta)), "w") as f:
        f.write(str(results))


def ensembling(lrate, mu, beta, epochs):
    n = 4
    best = 2
    networks = []
    for i in range(n):
        print(i)
        r = run_one_train([10, 20, 2], sigmoid, euclideanLoss, lrate, mu, beta, epochs, 32)
        print(r[0])
        networks.append(r)
    networks.sort()

    val, train, test = read_cup()
    total_mee = 0.
    for x,y in test:
        avg = np.zeros(2)
        for i in range(best):
            avg += networks[i][1].feed_forward(x)
        avg /= best
        total_mee += euclideanLoss.function(avg, y)
    return total_mee / len(test), networks




epochs = 2000
batch_size = 32

lrate = 0.1
mu = 0.001
beta = 0
act_fun = sigmoid
loss = euclideanLoss

size_list = [10, 20, 2]


#grid_search(hidden, act_fun, loss, epochs, batch_size)
#val_loss, net = run_one_train(size_list, act_fun, euclideanLoss, lrate, mu, beta, epochs, batch_size, debug=True)

res, nets = ensembling(lrate, mu, beta, 1500)
print(res)

with open(f"models/networks_ensemble_{res}.pkl", "wb") as f:
    pickle.dump(nets, f)
