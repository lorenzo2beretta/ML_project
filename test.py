from network import *
from gradient_descent import *
from utility import *


lrate = 0.5
epochs = 1000
size_list = [2, 6, 2]
network = Network(size_list, reLU, softMax)

data = []
for i in range (100):
    x = np.random.randint(2, size = 2)
    data.append((x, [x[0] * x[1], 1 - x[0] * x[1]]))

algo = GradientDescent(crossEntropy, lrate, epochs, network)
algo.train(data)

print(network.feed_forward(np.array([0, 1])))
print(network.feed_forward(np.array([1, 0])))
print(network.feed_forward(np.array([0, 0])))
print(network.feed_forward(np.array([1, 1])))
