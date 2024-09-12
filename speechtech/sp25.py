import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10000)
#math as in text
m = 2595 * np.log10(1 + x/700)

plt.plot(x,m)
plt.show()
