import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#make a complex wave
x = np.sin(np.arange(0,200*np.pi,.1))
z = np.sin(3*np.arange(0,200*np.pi,.1))
x = x + z

#build regression data
d = np.zeros([len(x)-4,5])
d[:,0] = x[:-4]
d[:,1] = x[1:-3]
d[:,2] = x[2:-2]
d[:,3] = x[3:-1]
d[:,4] = x[4:]

#do autoregression
r = linear_model.LinearRegression()
r.fit(d[:,:4],d[:,4])

#calculate predicted values for
#all other values
y = np.zeros([len(x),1])
y[:4,0] = x[:4]
for i in range(4,len(y)):
	y[i,0] = r.predict(y[i-4:i,0].reshape(1,-1))

#display first 1000 values
plt.plot(x[:1000])
plt.plot(y[:1000,0])
plt.show()

#print error
print(sum(abs(y[:,0]-x)))
