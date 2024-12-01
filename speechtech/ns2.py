import torch as t
import numpy as np
import matplotlib.pyplot as plt

#initialize results
allres = []
#initialize weights
w1 = t.tensor([[[1.,1.,1.]]])
w2 = t.tensor([[[1.,1.,1.]]])

#range of values to map from
therange = t.arange(-5,5,.1)


#iterate
for i in therange:
	inp = t.tensor([[[i,i,i]]])
	#the math
	res = t.tanh(t.conv1d(inp,w1)) * \
		t.sigmoid(t.conv1d(inp,w2))
	allres.append(np.array(res)[0][0])

#plot results
plt.plot(therange,allres)
plt.show()

