# question 3
from matplotlib import pyplot as plt
import numpy as np

X = np.arange(8*8).reshape(8, 8) % 2
for i in range(0, X.shape[0], 2):
    X[i] = (X[i] + 1) % 2

plt.title('Chess Board Pattern')
plt.imshow(X, cmap='grey')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
plt.tight_layout()
plt.show()
