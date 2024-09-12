class Arc:
	#initialize an arc
	def __init__(self,p,n,t,b,w):
		self.p = int(p)
		self.n = int(n)
		self.t = t
		self.b = b
		self.w = float(w)
	#display an arc
	def __repr__(self):
		res = str(self.p) + ' -> '
		res += str(self.n) + ', '
		res += self.t + ':'
		res += self.b
		res += ', w = ' + str(self.w)
		return res
