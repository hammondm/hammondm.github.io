import numpy as np
import matplotlib.pyplot as plt

#a sine wave
x = np.sin(np.arange(0,np.pi*4,.1))

mu = 255

#math as in text
numerator = np.log(1+(mu*np.abs(x)))
denominator = np.log(1+mu)
y = np.sign(x)*(numerator/denominator)

#plot
plt.plot(x)
plt.plot(y)
plt.show()
