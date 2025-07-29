import numpy as np
from sklearn.tree import \
	DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt

#input
x = np.arange(0,100,.1)
#output
y = x**3/10000
#add random noise to output
jitter = np.random.rand(len(x))-.5
jitter *= 3
y += jitter
#reshape input
x = x.reshape(-1,1)
#create tree
r = DecisionTreeRegressor(max_depth=3)
r.fit(x,y)
#predict
y2 = r.predict(x)

#plot
plt.subplot(2,1,1)
plt.plot(x,y,'b')
plt.plot(x,y2,'b')
plt.subplot(2,1,2)
plot_tree(r,label='none')

plt.show()

