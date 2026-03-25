import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

PATH = '../media/BostonHousing.csv'

Xy = np.genfromtxt(
    PATH, 
    delimiter=',',
    skip_header=1,
    dtype=np.float64,
    converters = {3: lambda s: float(s[1: -1])}
)

# Question 4.1 RM (column 5) and LSTAT (column 12) are the most correlated.
Xy = np.column_stack((Xy[:, 5], Xy[:, 12], np.ones(shape=Xy.shape[0]), Xy[:, -1]))
print(Xy[:20])
test = Xy[-100:]
train = Xy[:-100]
X, y = train[:, :-1], train[:, -1]
_X, _y = test[:, :-1], test[:, -1]

def ridge_beta(X, y,  lmb):
    p = X.shape[1]
    A = X.T @ X + lmb * np.eye(p)
    B = X.T @ y
    return np.linalg.solve(A, B)

def rmse(X, y, lmd):
    beta = ridge_beta(X, y, lmd) 
    y_pred = X @ beta
    return np.sqrt(np.mean((y - y_pred) ** 2))

# lambda
lmb = np.arange(-10, 20)
errors = [rmse(_X, _y, l) for l in  lmb]

plt.plot(lmb, errors)
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.title('Ridge Regression RMSE vs λ')
plt.grid(True)
plt.tight_layout()
plt.savefig('../media/ridgePlot.png')
plt.show()
