import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

#how many bits
y = 5
#make a wave
x = np.sin(np.arange(1,4*np.pi,.1))
#make the stepped wave by rounding
z = y * x
z = z.round()
z = z/y
#plot both
plt.plot(x,'b')
plt.plot(z,'b')
plt.show()
