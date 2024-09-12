#working out covariance

import numpy as np
from scipy.stats import \
	multivariate_normal as mn

#two random sets
x1 = np.array([2,7,4,4,5,3])
x2 = np.array([1,5,3,2,6,4])

#calculate means
mean1 = x1.mean()
print(f'mean1 = {mean1}')
mean2 = x2.mean()
print(f'mean2 = {mean2}')

#calculate covariance
cov = np.cov(x1,x2,bias=True)
print(cov)

#standard deviation
s1 = np.mean((x1 - mean1)**2)
s2 = np.mean((x2 - mean2)**2)

print(f's1 = {s1}')
print(f's2 = {s2}')
s12 = np.mean((x1 - mean1)*(x2 - mean2))
print(f's12 = {s12}')

#probability density function
p = mn.pdf(
	x=(2,2),
	mean=(x1.mean(),x2.mean()),
	cov=cov
)
print(f'p = {p}')

#calculate rho
rho = cov[0,1]/(np.sqrt(s1) * np.sqrt(s2))
print(f'rho = {rho}')

#pdf by hand
def pdf(x,y,xs,ys,xm,ym,r):
	xss = np.sqrt(xs)
	yss = np.sqrt(ys)
	left = 1/(2*np.pi*xss*yss*np.sqrt(1-r**2))
	rightpow = - 1/(2*(1-r**2))
	rightpow *= (((x-xm)/xss)**2 + \
		((y-ym)/yss)**2) - \
		2*r*((x-xm)/xss)*((y-ym)/yss)
	right = np.e**rightpow
	return left*right

#print result
p2 = pdf(2,2,s1,s2,mean1,mean2,rho)
print(f'p2 = {p2}')
