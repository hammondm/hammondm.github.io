import numpy as np
import matplotlib.pyplot as plt

#sequence of numbers from 0 to 30
x = np.arange(0,30,.1)
#apply sine function
s = np.sin(x)
#plot
plt.plot(s)
plt.show()