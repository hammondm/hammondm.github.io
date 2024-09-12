import numpy as np

def dft(x):
	#make a sequence of the same length
	r = np.arange(len(x))
	#virtually exactly the equation
	res = [
		(np.e**(-1j*2*np.pi*i*r/len(r))*x).sum()
			for i in r
	]
	return res
