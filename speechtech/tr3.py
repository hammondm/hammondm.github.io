import numpy as np
import numpy.matlib

def dtw(t,r):
	#convert to numpy arrays
	ta = np.array([t])
	ra = np.array([r])
	#get the sizes
	N = ta.size
	M = ra.size
	#expand to size of second array
	tax = np.matlib.repmat(ta,M,1)
	#expand to size of first array
	rax = np.matlib.repmat(ra.transpose(),1,N)
	#square the difference
	d = (tax-rax)**2
	#initialize matrix
	D = np.zeros(d.shape)
	#initialize first cell
	D[0,0] = d[0,0]
	#initialize first column
	for m in range(1,M):
		D[m,0] = d[m,0] + D[m-1,0]
	#initialize first row
	for n in range(1,N):
		D[0,n] = d[0,n] + D[0,n-1]
	#go through remaining cells
	for n in range(1,N):
		for m in range(1,M):
			#find best value
			D[m,n] = d[m,n] + min(
				[D[m,n-1],D[m-1,n-1],D[m-1,n]]
			)
	Dist = D[M-1,N-1]
	path = [(M-1,N-1)]
	#trace path backward
	while path[-1] != (0,0):
		left = path[-1][0]
		right = path[-1][1]
		if left == 0:
			path.append((left,right-1))
		elif right == 0:
			path.append((left-1,right))
		else:
			loc = np.argmin([
				D[left-1,right],
				D[left,right-1],
				D[left-1,right-1]
			])
			if loc == 0:
				path.append((left-1,right))
			elif loc == 1:
				path.append((left,right-1))
			else:
				path.append((left-1,right-1))
	#return values and path
	pairs = []
	for p in path:
		pair = (r[p[0]],t[p[1]])
		pairs.append(pair)
	pairs.reverse()
	return pairs,path

#test
if __name__ == '__main__':
	x = np.array([3,5,2])
	y = np.array([2,4,7,1])
	print(dtw(x,y))
