import numpy as np

#sigmoid function
def sigmoid(x): return 1/(1+np.exp(-x))

#initial variables
slope = .4
intercept = .8
r = .1
iterations = 100
samples = 100
w = .5
b = .2

#random inputs and targets
x = np.random.randint(0,100,samples)
x = x.astype(float)
x /= 100
y = x*slope + intercept
y += (np.random.rand(samples)-.5)/100
y = sigmoid(y)

print(f'target w: {slope}, b: {intercept}')

print(f'initial w = {w}, b = {b}')

#iterate
for j in range(iterations):
	#over all items
	for i in range(len(x)):
		#calculate current output
		res = sigmoid(x[i]*w + b)
		#update
		w += r * (y[i]-res) * res * (1-res) * x[i]
		b += r * (y[i]-res) * res * (1-res) * 1

print(f'resulting w = {w}, b = {b}')
