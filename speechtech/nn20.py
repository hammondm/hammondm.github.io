import numpy as np
import matplotlib.pyplot as plt

#how many data points
r = 50

#encoding function
def enc(pos,vsize):
	pe = np.zeros(vsize)
	for i in range(0,vsize,2):
		#sine values
		pe[i] = np.sin(
			pos/(10000**((2*i)/vsize))
		)
		#cosine values
		pe[i+1] = np.cos(
			pos/(10000**((2*i)/vsize))
		)
	return pe

#calculate values
res = []
for i in range(r):
	val = enc(i,4)
	res.append(val)

#plot
plt.plot(range(r),res)
plt.show()
