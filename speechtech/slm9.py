#HMMs in python
#
#Mike Hammond, U. of Arizona, 6/2021

import numpy as np

class HMM:
	'''Hidden Markov Model
	'''
	def __init__(self,numstates,alphabet):
		#transition matrix with random values
		tm = np.random.rand(numstates,numstates)
		#convert to probabilities
		tm = tm / np.asmatrix(tm.sum(axis=1)).transpose()
		self.tm = tm
		#lookup tables for letters
		self.a2i = {letter:index for 
			index,letter in enumerate(alphabet)}
		self.i2a = {index:letter for 
			index,letter in enumerate(alphabet)}
		#emissions matrix with random values
		em = np.random.rand(numstates,len(alphabet))
		#convert to probabilities
		em = em / np.asmatrix(em.sum(axis=1)).transpose()
		self.em = em
		#random start probabilities
		start = np.random.rand(numstates)
		start /= start.sum()
		self.start = start
	def forward(self,s):
		'''calculate forward probabilities

		args:
			s	list of characters

		returns:
			prob	total probability
			grid	accumulated probability grid
		'''
		#make matrix
		nodecount = self.tm.shape[0]
		grid = np.zeros([nodecount,len(s)])
		#go through the input letter by letter
		for letpos in range(len(s)):
			#get letter index
			letidx = self.a2i[s[letpos]]
			#get probs to the left
			#if we're in the first col, use start probs
			if letpos == 0:
				prev = self.start
			else:
				prev = grid[:,letpos-1]
			#go row by row, filling in values
			#sum for all states
			for node in range(nodecount):
				val = 0
				if letpos == 0:
					val = prev[node] * self.em[node,letidx]
				else:
					for n2 in range(nodecount):
						curval = prev[n2]
						curval *= self.tm[n2,node]
						curval *= self.em[node,letidx]
						val += curval
				grid[node,letpos] = val
		return grid[:,-1].sum(axis=0),grid
	def viterbi(self,s):
		'''Viterbi

		args:
			s	list of characters

		returns
			res	a list of states
		'''
		nodecount = self.tm.shape[0]
		#make matrix
		grid = np.zeros([nodecount,len(s)])
		best = np.zeros([nodecount,len(s)],dtype=int)
		#go through the input letter by letter
		for letpos in range(len(s)):
			#get letter index
			letidx = self.a2i[s[letpos]]
			#get probs to the left
			#if we're in the first col, use start probs
			if letpos == 0:
				prev = self.start
			else:
				prev = grid[:,letpos-1]
			#go row by row, filling in values
			#sum for all states
			for node in range(nodecount):
				val = []
				if letpos == 0:
					grid[node,letpos] = prev[node] * self.em[node,letidx]
				else:
					for n2 in range(nodecount):
						curval = prev[n2]
						curval *= self.tm[n2,node]
						curval *= self.em[node,letidx]
						val.append(curval)
					#keeps track of the max value
					grid[node,letpos] = max(val)
					#keeps track of where that max came from
					best[node,letpos] = val.index(max(val))
		#reads back the best path
		last = grid.argmax(axis=0)[-1]
		res = [last]
		i = len(s) - 1
		while i >= 1:
			res.append(best[res[-1],i])
			i -= 1
		res.reverse()
		return grid[:,-1].max(),res
	def backward(self,s):
		'''calculate backward probabilities

		args:
			s	list of characters

		returns:
			prob	total probability
			grid	accumulated probability grid
		'''
		nodecount = self.tm.shape[0]
		#make matrix
		grid = np.zeros([nodecount,len(s)])
		#rightmost column
		grid[:,-1] = 1
		cols = len(s) - 2
		i = cols
		#go through columns from R to L
		while i >= 0:
			nextletcode = self.a2i[s[i+1]]
			j = 0
			#go through rows top down
			while j < nodecount:
				cellval = 0
				k = 0
				while k < nodecount:
					val = 1
					val *= grid[k,i+1]
					val *= self.tm[j,k]
					val *= self.em[k,nextletcode]
					k += 1
					cellval += val
				grid[j,i] = cellval
				j += 1
			i -= 1
		return grid
	def train(self,ss):
		'''Baum-Welch

		Updates all probabilities based on a list of
		training items. This does a single iteration.

		args
			ss		a list of strings
		'''
		gammas = []
		xis = []
		r = len(ss)
		for s in ss:
			p,fg = self.forward(s)
			bg = self.backward(s)
			#make gamma (delta for that other guy)
			num = fg*bg
			gamma = num/num.sum(axis=0)
			gammas.append(gamma)
			#make xi (gamma for that other guy)
			numstates = self.tm.shape[0]
			xi = np.zeros([numstates,numstates,len(s)-1])
			for t in range(len(s)-1):
				#tpluslet = self.a2i[s[t]]
				tpluslet = self.a2i[s[t+1]]
				for i in range(numstates):
					for j in range(numstates):
						val = fg[i,t] * self.tm[i,j] * \
							bg[j,t+1] * self.em[j,tpluslet]
						xi[i,j,t] = val
			for t in range(len(s)-1):
				quadsum = sum(sum(xi[:,:,t]))
				xi[:,:,t] /= quadsum
			xis.append(xi)
		#UPDATE
		#start probabilities
		startprobs = np.zeros(numstates)
		for i in range(numstates):
			for gamma in gammas:
				startprobs[i] += gamma[i,0]
			startprobs[i] /= len(gammas)
		self.start = startprobs
		#transitions
		for i in range(numstates):
			for j in range(numstates):
				num = 0
				for xi in xis:
					#for t in range(xi.shape[2]-1):
					for t in range(xi.shape[2]):
						num += xi[i,j,t]
				denom = 0
				for gamma in gammas:
					for t in range(gamma.shape[1]-1):
						denom += gamma[i,t]
				self.tm[i,j] = num/denom
		#emissions
		for i in range(numstates):
			for k in range(len(self.a2i)):
				num = 0
				for g in range(len(gammas)):
					gamma = gammas[g]
					s = ss[g]
					for t in range(gamma.shape[1]):
						letnum = self.a2i[s[t]]
						if k == letnum:
							num += gamma[i,t]
				denom = 0
				for gamma in gammas:
					for t in range(gamma.shape[1]):
						denom += gamma[i,t]
				self.em[i,k] = num/denom

#test
if __name__ == '__main__':
	lets = ['a','b']

	print('t1')
	hmm = HMM(1,lets)
	hmm.tm = np.array([
		[1]
	])
	hmm.start = np.array(
		[1.0]
	)
	hmm.em = np.array([
		[.3,.7]
	])
	p,fg = hmm.forward(['a','a','b','b'])
	print(p,'\n')
	hmm.train([['a','a','b','b']])
	p,fg = hmm.forward(['a','a','b','b'])
	print(p)

	print('\nt2')
	hmm = HMM(2,lets)
	hmm.tm = np.array([
		[.1,.9],
		[.1,.9]
	])
	hmm.start = np.array(
		[1.0,0.0]
	)
	hmm.em = np.array([
		[1.0,0.0],
		[0.0,1.0]
	])
	p,fg = hmm.forward(['a','a','b','b'])
	print(p,'\n')
	hmm.train([['a','a','b','b']])
	p,fg = hmm.forward(['a','a','b','b'])
	print(p)

	print('\nt3')
	hmm = HMM(3,lets)
	hmm.start = np.array(
		[1.0,0.0,0.0]
	)
	hmm.tm = np.array([
		[0,.9,.1],
		[1,0,0],
		[1,0,0]
	])
	hmm.em = np.array([
		[1,0],
		[.2,.8],
		[.7,.3]
	])
	p,fg = hmm.forward(['a','b','a'])
	print(p,'\n')
	hmm.train([['a','b','a']])
	p,fg = hmm.forward(['a','b','a'])
	print(p)

