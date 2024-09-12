import numpy as np
import matplotlib.pyplot as plt

#activation functions
def sig(x): return np.exp(x)/(1+np.exp(x))
def lin(x): return x
#derivative versions
def dsig(y): return sig(y)*(1-sig(y)) 
def dlin(x): return 1

class NN:
	def __init__(self,layers,acts):
		self.layers = layers
		self.acts = acts
		self.weights = []
		self.biases = []
		#initialize weights and biases
		for i in range(len(layers)-1):
			self.weights.append(
				np.random.randn(layers[i+1],layers[i])
			)
			self.biases.append(
				np.random.randn(layers[i+1],1)
			)
	def forward(self,x):
		#copy input
		a = np.copy(x)
		#output of each level
		z_s = []
		#post-activation output
		a_s = [a]
		#layer by layer
		for i in range(len(self.weights)):
			#get activation function
			actfunc = self.acts[i]
			#output without activation
			z_s.append(
				self.weights[i].dot(a) + 
				self.biases[i]
			)
			#apply activation
			a = actfunc(z_s[-1])
			a_s.append(a)
		return z_s,a_s
	def backward(self,y,z_s,a_s):
		dw = []  # dC/dW
		db = []  # dC/dB
		#delta = dC/dZ
		deltas = [None] * len(self.weights)
		#last layer error first
		if self.acts[-1] == lin: dactfunc = dlin
		else: dactfunc = dsig
		deltas[-1] = ((y-a_s[-1]) * \
			(dactfunc)(z_s[-1]))
		#backprop (start from last layer):
		for i in reversed(range(len(deltas)-1)):
			if self.acts[i] == lin: dactfunc = dlin
			else: dactfunc = dsig
			deltas[i] = self.weights[i+1].T.dot(
				deltas[i+1]
			)*dactfunc(z_s[i])
		#go in batches
		bsize = y.shape[1]
		#new biases
		db = [d.dot(np.ones((bsize,1)))/ \
			float(bsize) for d in deltas]
		#new weights
		dw = [d.dot(a_s[i].T)/float(bsize) \
			for i,d in enumerate(deltas)]
		#return derivitives
		return dw,db
	def train(self,x,y,bsize,epochs,lr):
		losses = []
		#update weights and biases based on output
		for e in range(epochs): 
			i = 0
			if e % 100 == 0: print(f'epoch: {e:>4}')
			while(i < len(y)):
				#put items in batches
				x_batch = x[i:i+bsize]
				y_batch = y[i:i+bsize]
				i = i + bsize
				#get values for current batch
				z_s,a_s = self.forward(x_batch)
				#backprop
				dw,db = self.backward(y_batch,z_s,a_s)
				#update weights
				self.weights = [w+lr*dweight for \
					w,dweight in zip(self.weights,dw)]
				#update biases
				self.biases = [w+lr*dbias for \
					w,dbias in zip(self.biases,db)]
				#current loss (sqrt(sum(x**2)))
				loss = np.linalg.norm(a_s[-1]-y_batch)
				losses.append(loss)
		return losses

if __name__=='__main__':
	#network
	nn = NN([1,10,10,1],acts=[sig,sig,lin])
	#random input
	X = 5*np.random.rand(1000).reshape(1,-1)
	#simple output
	y = X**2 + 4
	#train
	losses = nn.train(
		X,
		y,
		epochs=2000,
		bsize=100,
		lr=.01
	)
	#plot
	_,a_s = nn.forward(X)
	plt.subplot(2,1,1)
	plt.subplots_adjust(hspace=.4)
	plt.scatter(
		X.flatten(),
		y.flatten(),
		s=4,
		c='r'
	)
	plt.scatter(
		X.flatten(),
		a_s[-1].flatten(),
		s=4,
		c='b'
	)
	plt.title(
		'true (red) and predicted (blue) values'
	)
	plt.subplot(2,1,2)
	plt.plot(losses)
	plt.title('loss')
	plt.show()

