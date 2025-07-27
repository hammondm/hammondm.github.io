import numpy as np
import matplotlib.pyplot as plt

#how sparse is the sampling
y = 1.3
x = np.sin(np.arange(0,4*np.pi,y))
z = np.sin(np.arange(0,4*np.pi,.0001))
#plot curve
plt.plot(np.arange(0,4*np.pi,.0001),z,'b')
#plot approximation
plt.plot(np.arange(0,4*np.pi,y),x,'b')
#add vertical lines for visibility
plt.stem(
	np.arange(0,4*np.pi,y),
	x,
	linefmt='b',
	basefmt='b'
)
plt.show()

