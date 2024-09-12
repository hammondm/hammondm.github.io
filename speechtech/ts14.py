import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

#number of clusters
k = 3

#number of samples per distribution
ns = 300

#cluster means
targetmeans = [
	[8.4,8.2],
	[2.4,5.4],
	[6.4,2.4]
]

#positive semidefinite convariance matrices
targetcovs = []
for s in range(len(targetmeans)):
	targetcovs.append(make_spd_matrix(2))

X = []
for mean,cov in zip(targetmeans,targetcovs):
	x = np.random.multivariate_normal(
		mean,
		cov,
		ns
	)
	X += list(x)
  
X = np.array(X)
np.random.shuffle(X)

#initialize clusters and weights
weights = np.ones((k))/k
means = np.random.choice(
	X.flatten(),
	(k,X.shape[1])
)
print(f'initial means\n{means}')
print(f'initial weights\n{weights}')

#positive semidefinite convariance matrix 
cov = []
for i in range(k):
  cov.append(make_spd_matrix(X.shape[1]))
cov = np.array(cov)
print(f'initial covariance\n{cov}')

better = []
#30 iterations
for step in range(30):
	print(f'step {step+1}')
	likelihood = []
	#calculate probabilities
	for j in range(k):
		likelihood.append(
			multivariate_normal.pdf(
				x=X,
				mean=means[j],
				cov=cov[j]
			)
		)
	likelihood = np.array(likelihood)
	better.append(
		likelihood.T.dot(weights).sum()
	)
	b = []
	#calculate b
	for j in range(k):
		denom = np.sum(
			[likelihood[i]*weights[i] \
				for i in range(k)],
			axis=0
		)
		b.append((likelihood[j]*weights[j])/denom)
		#update mean and covariance
		num = np.sum(
			b[j].reshape(len(X),1)*X,
			axis=0
		)
		means[j] = num/np.sum(b[j])
		cov[j] = np.dot(
			(b[j].reshape(len(X),1) * \
				(X - means[j])).T,
			(X - means[j])) / (np.sum(b[j]))
		#update weights
		weights[j] = np.mean(b[j])
	print(f'means\n{means}')
	print(f'weights\n{weights}')
	print(f'covariance\n{cov}')

print(f'\ntarget means\n{targetmeans}')

plt.plot(better)
plt.show()

