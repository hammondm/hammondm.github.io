import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

#make a sine wave
x = np.sin(np.arange(0,4*np.pi,.1))
#calculate differences
y = x[1:]-x[:-1]
x = x[:-1]
#plot both
plt.plot(x,'b')
plt.plot(y,'b')
plt.show()
