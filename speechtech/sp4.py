import numpy as np
import matplotlib.pyplot as plt

#sequence from 0 to 6pi
x = np.arange(0,6*np.pi,.1)
#shift 1/4 cycle to the "right"
y = x + np.pi/2
#plot both
plt.plot(np.sin(x),'b')
plt.plot(np.sin(y),'b')
plt.show()
