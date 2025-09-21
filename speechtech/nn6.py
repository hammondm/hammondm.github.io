import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['savefig.dpi'] = 300

def sigmoid(x): return 1/(1+np.exp(-x))

a = np.arange(-10,10,.1)
b = sigmoid(a)

plt.plot(a,b)
plt.show()
