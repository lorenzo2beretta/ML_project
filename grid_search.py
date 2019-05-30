from network import *
from gradient_descent import *
from utility import *
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pickle
import itertools

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
    n = 20
    best = 10
    size_list = [55, 20, 2]
    networks = []
    for i in range(n):
        print(i)
        r = run_one_train(size_list, sigmoid, euclideanLoss, lrate, mu, beta, epochs, 32)
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




epochs = 1000
batch_size = 32

lrate = 0.1
mu = 0.001
beta = 0
act_fun = sigmoid
loss = euclideanLoss

size_list = [10, 20, 2]


val, train, test = read_cup()


# tot_mee = 0.
# for _ in range(10):
#     val_loss, net = run_one_train(size_list, act_fun, loss, lrate, mu, beta, epochs, batch_size, debug=True)
#     tot_mee += net.avg_loss(test, loss)
#
# print(tot_mee / 10)

# res, nets = ensembling(lrate, mu, beta, 1000)
# print(res)
#
# with open(f"models/networks_ensemble_{res}.pkl", "wb") as f:
#     pickle.dump(nets, f)

ten = "models/networks_ensemble_1.0850775830603638.pkl"
ff = "models/networks_ensemble_1.0627232513968625.pkl"
with open(ff, "rb") as f:
    nets = pickle.load(f)

with open(ten, "rb") as f:
    nets2 = pickle.load(f)

# diffs = []

#
# n_avg = 15
# tot_mee = 0.
# for x,y in test:
#     avg = np.zeros(2)
#     for i in range(n_avg):
#         avg += nets[i][1].feed_forward(x)
#     avg /= n_avg
#     diffs.append(avg-y)
#     tot_mee += euclideanLoss.function(avg, y)
#
# diffs = np.array(diffs)
#
# print(tot_mee / len(test), np.mean(diffs, axis=0), np.std(diffs, axis=0))


test_set_in = []
test_set_10 = []
with open("cup/ML-CUP18-TS.csv", "r") as infile:
    reader = csv.reader(infile, delimiter=",")
    for row in reader:
        vars = [float(x) for x in row[1:11]]
        data = np.array(vars+[vars[i]*vars[j] for i, j in itertools.combinations(range(10),2) ])
        test_set_in.append(data)
        test_set_10.append(np.array(vars))

n_avg = 15
results = []
results_10 = []
for x in test_set_in:
    avg = np.zeros(2)
    for i in range(n_avg):
        avg += nets[i][1].feed_forward(x)
    cur_res = avg / n_avg
    results.append(cur_res)

n_avg = 10
for x in test_set_10:
    avg = np.zeros(2)
    for i in range(n_avg):
        avg += nets2[i][1].feed_forward(x)
    cur_res = avg / n_avg
    results_10.append(cur_res)

results = np.array(results)
results_10 = np.array(results_10)
diff = results-results_10
print(np.linalg.norm(diff,axis=1).mean())

plt.scatter(results[:,0],results[:,1])
plt.show()

with open("results_ML-CUP18-TS.csv", "w") as outfile:
    writer = csv.writer(outfile, delimiter=",")
    for i,r in enumerate(results):
        writer.writerow([i+1, r[0], r[1]])
