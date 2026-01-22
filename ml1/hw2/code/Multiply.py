# question 4
from matplotlib import pyplot as plt
import numpy as np

a = np.matrix('1 0 1; 2 1 1; 0 1 1; 1 1 2')
b = np.matrix('1 2 1; 2 3 1; 4 2 2')

# test to see if I did it right
print('a = \n', a, '\nb = \n', b)

# dot product both matrices
ab = np.dot(a,  b)
print('AB = ', '\n', ab)