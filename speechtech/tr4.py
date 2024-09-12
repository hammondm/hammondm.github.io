import numpy as np

#sqrt of sum of squared diffs
def sssd(a,b):
	res = np.sqrt(sum((a - b)**2))
	return res

def dtwmin(ta,ra,cost=sssd):
	N = ta.shape[1]
	M = ra.shape[1]
	d = np.zeros([N,M])
	#matrix for current costs
	for n in range(N):
		for m in range(M):
			d[n,m] = cost(ta[:,n],ra[:,m])
	D = np.zeros(d.shape)
	#first cell
	D[0,0] = d[0,0]
	#first column
	for n in range(1,N):
		D[n,0] = d[n,0] + D[n-1,0]
	#first row
	for m in range(1,M):
		D[0,m] = d[0,m] + D[0,m-1]
	#all remaining cells
	for n in range(1,N):
		for m in range(1,M):
			D[n,m] = d[n,m] + min([
				D[n-1,m],
				D[n-1,m-1],
				D[n,m-1]
			])
	#lower right val
	return D[N-1,M-1]

if __name__ == '__main__':
	x = np.array([
		[1,5,6],
		[2,-3,4]
	])
	y = np.array([
		[4,-1],
		[5,3]
	])
	print(dtwmin(x,y))
