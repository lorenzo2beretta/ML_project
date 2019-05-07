from network import *
from gradient_descent import *

def loss_fun(a, b):
    return np.dot(a - b, a - b)

def loss_der(a, b):
    return 2 * (a - b)

loss = DiffFunction(loss_fun, loss_der)
lrate = 0.01
epochs = 1000

def reLU_fun(x):
    x[x <= 0] = 0
    return x
def reLU_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

reLU = DiffFunction(reLU_fun, reLU_der)
id = DiffFunction(lambda x: x, lambda x: 1)
size_list = [2, 4, 1]
network = Network(size_list, reLU, id)

data = []

for i in range (400):
    x = np.random.randint(2, size = 2)
    data.append((x, [x[0] and x[1]]))

algo = GradientDescent(loss, lrate, epochs, network)
algo.train(data)

res = network.feed_forward(np.array([0, 1]))
print(res)
