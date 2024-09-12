import numpy as np

#probability density function
def prob(data,mean,variance):
	s1 = 1/(np.sqrt(2*np.pi*variance))
	s2 = np.exp(
		-(np.square(data-mean)/(2*variance))
	)
	return s1 * s2

#number of items for each distribution
ns = 300

#means and variances
mu1,sigma1 = -2,0.9
mu2,sigma2 =  4,1.8
mu3,sigma3 =  0,2.3

#random samples for each
x1 = np.random.normal(mu1,np.sqrt(sigma1),ns)
x2 = np.random.normal(mu2,np.sqrt(sigma2),ns)
x3 = np.random.normal(mu3,np.sqrt(sigma3),ns)

#assemble and randomize
X = np.hstack([x1,x2,x3])
np.random.shuffle(X)

#how many clusters
k = 3
#initial values/guesses
weights = np.ones((k))/k
means = np.random.choice(X,k)
variances = np.random.random_sample(size=k)

#EM loop
for step in range(50):
	print(f'step {step+1}')
	print(f'\tweights = {weights}')
	print(f'\tmeans = {means}')
	print(f'\tvariances = {variances}')
	likelihood = []
	#current probs
	for j in range(k):
		likelihood.append(
			prob(
				X,
				means[j],
				np.sqrt(variances[j])
			)
		)
	likelihood = np.array(likelihood)
	b = []
	for j in range(k):
		denom = np.sum(
			[likelihood[i]*weights[i] \
				for i in range(k)],
			axis=0
		)
		b.append(
			(likelihood[j]*weights[j])/denom
		)
		#update means and variances
		means[j] = np.sum(b[j] * X)/np.sum(b[j])
		num = np.sum(
			b[j] * np.square(X - means[j])
		)
		variances[j] = num / np.sum(b[j])
		#update weights
		weights[j] = np.mean(b[j])

print(f'\ntarget means = {(mu1,mu2,mu3)}')
print(
	'target variances = ' +
	f'{(sigma1,sigma2,sigma3)}'
)

