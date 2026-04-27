import numpy as np
from matplotlib import pyplot as plt 
from data import X, y, X_, y_

# architechture
p, h, l = X.shape[1], 8, y.shape[1]
biases = np.empty(2, dtype=object)
weights = np.empty(2, dtype=object)
biases[0], biases[1] = np.zeros((h, 1)), np.zeros((l, 1))
weights[0], weights[1] = np.zeros((h, p)), np.zeros((l, h))
L = len(weights) - 1

# network
def sigmoid(x):
    pass

def feedfoward(a, W, b, i=0):
    pass

def backpropagation(a_last, t,  W, b, i=0, gradient=[]):
    pass

def epoch(X, y, W, b, eta):
    for x, t in zip(X, y):
        gradient = backpropagation(x, t, W, b)
        for i in range(len(gradient)):
            b[i] -= eta * gradient[i][0]
            W[i] -= eta * gradient[i][1]

def rmse(y_pred, y):
    return np.sort(np.mean(np.mean((y_pred - y)**2, axis=1 )))

eta = 0.1

errors = []
for i in range(350):
    epoch(X, y, weights, biases, eta)
    y_pred = np.array([feedfoward(x, weights, biases) for x in X])
    errors.append(rmse(y_pred, y))

plt.title('Training Errors')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.tight_layout()
plt.plot(errors)
plt.show()
