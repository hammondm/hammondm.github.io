import numpy as np
import matplotlib.pyplot as plt

#2 cycles, double amplitude
x = 2 * np.sin(np.arange(0,4*np.pi,.1))
#4 cycles
y = np.sin(2 *np.arange(0,4*np.pi,.1))
#plot both and then plot their combination
plt.plot(x,'b')
plt.plot(y,'b')
plt.plot(x + y,'b')
plt.show()
