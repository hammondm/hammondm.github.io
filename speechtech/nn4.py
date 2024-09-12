import numpy as np
import matplotlib.pyplot as plt

#the function
def func(x): return x**2 + 3
#its derivative
def deriv(x): return 2*x

#initial variables
val = 1.
goal = 3.
r = .01

vals = []
losses = []
loss = func(goal) - func(val)
#iterate
while np.abs(loss) > .01:
	vals.append(val)
	losses.append(loss)
	#update value with derivative
	val += r * deriv(val) * loss
	#get loss
	loss = func(goal) - func(val)

#plot
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=.4)
plt.plot(vals)
plt.title('value of x')
plt.subplot(2,1,2)
plt.plot(losses)
plt.title('loss')
plt.show()
