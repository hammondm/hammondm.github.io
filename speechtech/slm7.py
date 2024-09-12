from slm6 import parsefst

def wc(t1,t2):
	f = {}
	arcs = set()
	#collect arcs and final states
	a1,f1 = parsefst(t1)
	a2,f2 = parsefst(t2)
	#two queues for states
	queue = [('0','0')]
	qx = [('0','0')]
	while len(qx) > 0:
		#look at current state pair
		q = qx.pop(0)
		q1 = q[0]
		q2 = q[1]
		#final weight is sum of final weights
		if q1 in f1 and q2 in f2:
			rho = f1[q1] + f2[q2]
			f[q] = rho
		#go through all arc combinations
		for arc1 in a1:
			for arc2 in a2:
				#if they match
				if arc1.b == arc2.t and \
						arc1.p == q1 and arc2.p == q2:
					#make a new state
					qprime = (arc1.n,arc2.n)
					if qprime not in queue:
						queue = [qprime] + queue
						qx = [qprime] + qx
					#sum arc weights
					arc = Arc(
						q,
						qprime,
						arc1.t,
						arc2.b,
						arc1.w + arc2.w
					)
					arcs.add(arc)
	return arcs,f
