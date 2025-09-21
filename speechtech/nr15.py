import math
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

def gelu(x):
	res = x * 1/2 * (1 + math.erf(x/(math.sqrt(2))))
	return res

v = [x/10 for x in range(-100,100)]
w = list(map(gelu,v))

plt.plot(v,w)
plt.show()

