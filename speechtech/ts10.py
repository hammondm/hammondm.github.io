import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3,3,.1)

#math as in text
def norm(x,mu,s):
	res = 1/(s*np.sqrt(2*np.pi)) * \
		np.e **(-1/2 * ((x-mu)/s)**2)
	return res

n = norm(x,0,1)

#plot
plt.plot(x,n)
plt.show()
