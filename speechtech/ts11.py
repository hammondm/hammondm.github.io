import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sss
from sklearn.mixture import GaussianMixture

#some random data
x = np.random.normal(10,5,1000)

#best fit of data as single gaussian
(mu,sigma) = sss.norm.fit(x)

#random data, two gaussians mixed together
a = np.random.normal(10,5,1000)
b = np.random.normal(30,5,1000)
c = np.hstack([a,b])

#best fit of two-gaussian data
(mu2,sigma2) = sss.norm.fit(c)

fig,axs = plt.subplots(1,3)

#histogram of the single gaussian data
n,bins,patches = axs[0].hist(x,60,density=1)
#best fit line
y = sss.norm.pdf(bins,mu,sigma)
axs[0].plot(bins,y,'b',linewidth=2)

#histogram of the two-gaussian data
n2,bins2,patches2 = \
	axs[1].hist(c,60,density=1)
#best fit line
y2 = sss.norm.pdf(bins2,mu2,sigma2)
axs[1].plot(bins2,y2,'b',linewidth=2)

#massage data for GMM
d = np.asmatrix(c)
d = d.T
#create GMM with 2 components
gmm = GaussianMixture(n_components=2)
#fit with GMM
gmm.fit(np.asarray(d))
x = np.linspace(np.min(d),np.max(d),len(d))
x = np.asmatrix(x)
x = x.T
fit = gmm.score_samples(np.asarray(x))

#histogram of the data
n,bins,patches = axs[2].hist(d,60,density=1)

#best fit line
y = sss.norm.pdf(bins,mu,sigma)
axs[2].plot(x,np.exp(fit),color='blue')

plt.show()

