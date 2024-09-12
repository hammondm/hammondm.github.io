import numpy as np

#softmax function
def softmax(x):
	e = np.exp(x)
	return e/e.sum()

#keys (indices)
k = np.array([
	[5,0,0],
	[0,5,0],
	[2,1,2]
])

dk = k.shape[1]

#query
q = np.array(
	[0,5,0]
)

#values (what we want)
v = np.array([7,8,9])

#find the best fit
res = q.dot(k.T)/np.sqrt(dk)

#softmax normalization
print(softmax(res))

#values weighted by fit
print(softmax(res)*v)

