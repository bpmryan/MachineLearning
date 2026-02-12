import numpy as np
from matplotlib import pyplot as plt

mean = [0, 0]
cov = [[1, 0.9], 
       [0.9, 1]]

rng = np.random.default_rng(0)
X, y = rng.multivariate_normal(mean, cov, 100).T

X = np.vstack((X, np.ones(shape=X.shape))).T

beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
m, b = beta

a = np.linspace(X.min(), X.max())
c = m*a + b

plt.title('Linear Regression in Two Dimensions')
plt.plot(X[:, 0], y, 'ko', alpha=0.5, label='Data')
plt.plot(a, c, 'b.-', label='Line of Bestfit')
plt.legend()
plt.tight_layout()

plt.show()
# plt.savefig('best_fit_plot.png')
# print("Plot saved as 'best_fit_plot.png'. Open this file in Windows to see the result.")