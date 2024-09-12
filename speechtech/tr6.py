import matplotlib.pyplot as plt
import numpy as np

count = 30
clusters = 2

#distance function
def dist(x,y): return np.sqrt(sum((x-y)**2))

#random input dimensions
x = np.random.uniform(0,100,count)
y = np.random.uniform(0,100,count)
#make points
points = np.vstack([x,y])

#random initial centroids
x = np.random.uniform(0,100,clusters)
y = np.random.uniform(0,100,clusters)
centroids = np.vstack([x,y])

#iterate 4 times
for k in range(4):
	c0 = []
	c1 = []
	#get closest points
	for i in range(points.shape[1]):
		d0 = dist(points[:,i],centroids[:,0])
		d1 = dist(points[:,i],centroids[:,1])
		if d0 < d1:
			c0.append(i)
		else:
			c1.append(i)

	#plot
	plt.subplot(2,2,k+1)

	plt.scatter(
		centroids[0,:],
		centroids[1,:],
		marker='X',
		s=60,
		color='black'
	)
	plt.scatter(
		points[0,c0],
		points[1,c0],
		marker='o',
		color='black'
	)
	plt.scatter(
		points[0,c1],
		points[1,c1],
		marker='s',
		color='black'
	)
	plt.title('# ' + str(k+1))
	#update centroids
	centroids[:,0] = points[:,c0].mean(axis=1)
	centroids[:,1] = points[:,c1].mean(axis=1)

plt.show()
