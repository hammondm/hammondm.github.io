import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

#sequence from 0 to 20
x = np.arange(0,20,.1)
plt.subplot(3,1,1)
#triple the frequency
plt.plot(np.sin(3 * x))
plt.subplot(3,1,2)
#double the frequency
plt.plot(np.sin(2 * x))
plt.subplot(3,1,3)
#the original
plt.plot(np.sin(x))
plt.show()
