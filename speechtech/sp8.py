import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

#parameters for a number distribution
mean = 0
std = 1 
num = 1000
#create distribution
w = np.random.normal(
  mean,std,size=num
)
#plot it
plt.plot(w)
plt.show()
