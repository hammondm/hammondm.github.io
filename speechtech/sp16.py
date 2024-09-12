import numpy as np
import matplotlib.pyplot as plt

#function for toy filter
def transfunc(x):
	if x < 1000: return x*.6 + 500
	else: return x*-.5 + 1600

#harmonics
x = np.arange(100,2001,100)
y = np.zeros(20)
z =  np.linspace(1000,800,20)

#draw harmonics
plt.subplot(3,1,1)
plt.vlines(x,y,z)
#draw both
xs = np.arange(0,2000)
ys = np.array(list(map(transfunc,xs)))
plt.plot(xs,ys)

#adjust harmonic amplitudes
for i,v in enumerate(x):
	val = transfunc(v)
	if val < z[i]: z[i] = val

#plot new harmonics
plt.subplot(3,1,2)
plt.vlines(x,y,z)
plt.plot(xs,ys)
plt.subplot(3,1,3)
plt.vlines(x,y,z)
plt.show()
