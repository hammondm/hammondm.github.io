import numpy as np

class LGU:
	#initialize weights
	e = np.array([])
	#set initial values
	def __init__(self,thresh=1,ins=e,ws=e):
		self.threshold = thresh
		self.inputs = ins
		self.weights = ws
	#calculate output
	def fwd(self):
		val = self.weights.dot(self.inputs)
		if val >= self.threshold: return 1
		else: return 0

#test
if __name__ == '__main__':
	#different test configuations
	pats = [
		('conj',np.array([1,1]),
			np.array([1,1]),2),
		('conj',np.array([1,1]),
			np.array([0,1]),2),
		('disj',np.array([1,1]),
			np.array([1,0]),1),
		('disj',np.array([1,1]),
			np.array([0,0]),1),
		('neg',np.array([-1]),np.array([1]),0),
		('neg',np.array([-1]),np.array([0]),0)
	]
	#go through test configurations
	for pat in pats:
		lgu = LGU(
			ws=pat[1],
			ins=pat[2],
			thresh=pat[3]
		)
		print(f'{pat[0]:>4}:' +
			f' {lgu.inputs} -> {lgu.fwd()}')
