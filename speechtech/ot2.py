import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

class SVM:
	#initialize
	def __init__(self,lr=0.001,lam=0.01):
		#learning rate
		self.lr = lr
		#width of zone vs. overlap weight
		self.lambdapar = lam
		#how many iterations
		self.n_iters = 100
		self.w = None
		self.b = None
	def fit(self,X,y):
		#get shape of data
		n_samples,nfeats = X.shape
		#convert labels to -1 and 1
		y_ = np.where(y <= 0,-1,1)
		#initialize weights
		self.w = np.zeros(nfeats)
		#initialize bias
		self.b = 0
		#optimize
		for _ in range(self.n_iters):
			#go through each item
			for idx,x_i in enumerate(X):
				#check if val is too high or low
				condition = y_[idx] * \
					(np.dot(x_i,self.w) - self.b) >= 1
				#adjust up
				if condition:
					self.w -= self.lr * \
						(2 * self.lambdapar * self.w)
				#adjust down
				else:
					self.w -= self.lr * (
						(2 * self.lambdapar * self.w) - 
							np.dot(x_i,y_[idx])
					)
					#adjust bias
					self.b -= self.lr * y_[idx]
	#get class of an element
	def predict(self,X):
		approx = np.dot(X,self.w) - self.b
		return np.sign(approx)

#get line
def hyperplane(x,w,b,offset):
	return (-w[0] * x + b + offset) / w[1]

#test
X,y = datasets.make_blobs(
	n_samples=20,
	n_features=2,
	centers=2,
	cluster_std=1.05,
	random_state=40
)
#convert labels to -1 and 1
y = np.where(y == 0,-1,1)
#initialize svm
clf = SVM()
#fit svm
clf.fit(X,y)
print(f'weights: {clf.w}, bias: {clf.b}')

#display
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X[:,0],X[:,1],marker="o")
x0_1 = np.amin(X[:,0])
x0_2 = np.amax(X[:,0])
#middle line
x1_1 = hyperplane(x0_1,clf.w,clf.b,0)
x1_2 = hyperplane(x0_2,clf.w,clf.b,0)
#bottom line
x1_1_m = hyperplane(x0_1,clf.w,clf.b,-1)
x1_2_m = hyperplane(x0_2,clf.w,clf.b,-1)
#top line
x1_1_p = hyperplane(x0_1,clf.w,clf.b,1)
x1_2_p = hyperplane(x0_2,clf.w,clf.b,1)
#plot lines
ax.plot([x0_1,x0_2],[x1_1,x1_2],"y--")
ax.plot([x0_1,x0_2],[x1_1_m,x1_2_m],"k")
ax.plot([x0_1,x0_2],[x1_1_p,x1_2_p],"k")
#dimensions of plot
x1_min = np.amin(X[:,1])
x1_max = np.amax(X[:,1])
ax.set_ylim([x1_min - 3,x1_max + 3])
#plot
plt.show()

