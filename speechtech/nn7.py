import numpy as np

#sigmoid function
def sigmoid(x): return 1/(1+np.exp(-x))

#input,input,target
data = np.array([
	[0,0,0],
	[0,1,1],
	[1,0,1],
	[1,1,1]	#change to [1,1,0] for XOR
])

#initial variables
r = 1.
iters = 5000
ws = np.array([.2,.2])
b = np.array([.4])

#initial output
print(f'initial ws = {ws}, b = {b}')
for i in range(data.shape[0]):
	row = data[i,:]
	res = sigmoid(row[:2].dot(ws) + b)
	print(f'{row}: {res}')

#iterate
for j in range(iters):
	for i in range(data.shape[0]):
		row = data[i,:]
		res = sigmoid(row[:2].dot(ws) + b)
		#update values
		ws += r * (row[2]-res) * \
			res * (1-res) * row[:2]
		b += r * (row[2]-res) * res * (1-res) * 1

#print results
print(
	f'ws = {ws}, b = {b} after {iters} updates'
)
for i in range(data.shape[0]):
	row = data[i,:]
	res = sigmoid(row[:2].dot(ws) + b)
	print(f'{row}: {res}')
