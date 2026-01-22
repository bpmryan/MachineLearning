from matplotlib import pyplot as plt
import numpy as np

# question 1
x = np.linspace(-np.pi, np.pi, 1000)    
f = x * (np.sin(x)**2)
g = -x * (np.sin(x)**2)

plt.title('Trignometric Functions')
plt.plot(x, f, 'bx-', label=r'$x\sin(x)^2$')
plt.plot(x, g, 'rx-', label=r'$x\sin^2(x)$')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.tight_layout()
plt.show()
