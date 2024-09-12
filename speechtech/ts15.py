from hmmlearn import hmm
import numpy as np

model = hmm.GaussianHMM(
	n_components=5,
	covariance_type="diag",
	n_iter=100
)

#how many features
K = 4
model.n_features = K 
#single input with length 100
X1 = np.random.randn(100,K)

#fit model with baum-welch
model.fit(X1)

#sample fitted model
X,Z = model.sample(20)

#print values
print(X)
#print states
print(Z)
