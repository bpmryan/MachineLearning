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
test = Xy[-100:]
train = Xy[:-100]
X, y = train[:, :-1], train[:, -1]
_X, _y = test[:, :-1], test[:, -1]

def ridge_beta(X, y,  lmb):
    p = X.shape[1]
    I = np.eye(p)
    return np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y

def rmse(X, y, _X, _y, lmb):
    beta = ridge_beta(X, y, lmb) 
    y_pred = _X @ beta
    return np.sqrt(np.mean((_y - y_pred) ** 2))

# lambda
lmb = np.arange(-10, 20)
errors = [rmse(X, y, _X, _y, l) for l in  lmb]

# Answer to question 3
best_lmb = lmb[np.argmin(errors)]
best_rmse = min(errors)
print(f"Best λ: {best_lmb}, RMSE: {best_rmse:.4f}")

plt.plot(lmb, errors)
plt.xlabel('λ')
plt.ylabel('RMSE')
plt.title('Ridge Regression RMSE vs λ')
plt.xticks(np.arange(-10, 20, 2))
plt.grid(True)
plt.tight_layout()
plt.savefig('../media/ridgePlot.png')
plt.show()

