import numpy as np

#set initial variables
slope = .4
intercept = .8
r = .1
iterations = 10
samples = 100
w = .6
b = .2

#create random input and targets
x = np.random.randint(0,100,samples)
x = x.astype(float)
x /= 100
y = x*slope + intercept
#add noise to targets
y += (np.random.rand(samples)-.5)/5

print(f'target w: {slope}, b: {intercept}')

print(f'initial w = {w}, b = {b}')

#do n iterations
for j in range(iterations):
	#go through all items
	for i in range(len(x)):
		#nudge as needed
		res = x[i]*w + b
		w += r * (y[i]-res) * x[i]
		b += r * (y[i]-res) * 1

print(f'resulting w = {w}, b = {b}')
