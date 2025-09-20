import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

#sequence from 0 to 26
x = np.sin(np.arange(0,26,.1))
#triple the amplitude
plt.plot(3 * x,'b')
#double the amplitude
plt.plot(2 * x,'b')
#the original
plt.plot(x,'b')
plt.show()
