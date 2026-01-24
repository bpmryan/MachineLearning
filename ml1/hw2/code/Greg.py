# question 5
from matplotlib import pyplot as plt
import numpy as np

# numbers to plug into series
i = np.arange(1, 101)

# series that numbers are plugged into
series = (4 * (-1)**(i + 1)) / ((2*i) - 1)

# result of each number
fn = np.cumsum(series)

# error curve
error = (np.pi - fn)**2

plt.title('The series of Greg (Gregory Series)')
plt.plot(i, fn, color="grey", label='Gregory Series')
plt.plot(i, error, color='red', label='Error curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()