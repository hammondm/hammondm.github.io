from slm2 import Arc

def myshortestpath(fst):
	#break into bits
	fst = fst.split('\n')
	arcs = []
	finalstates = []
	#go through the lines
	for line in fst:
		bits = line.split('\t')
		#add arcs
		if len(bits) == 5:
			a = Arc(*bits)
			arcs.append(a)
		#add final states
		elif len(line) > 0:
			finalstates.append(int(bits[0]))
	#extract all states from arcs
	fromstates = {arc.p for arc in arcs}
	tostates = {arc.n for arc in arcs}
	states = fromstates.union(tostates)
	#find the max weight
	maxweight = sum([arc.w for arc in arcs])
	maxweight *= 2
	#initialize tracking variables
	d = [0]
	r = [0]
	p = [[]]
	for i in range(1,len(states)):
		d.append(maxweight)
		r.append(maxweight)
		p.append([])
	queue = [0]
	#go through all states
	while len(queue) > 0:
		q = queue.pop(0)
		rprime = r[q]
		r[q] = maxweight
		#for each arc
		for arc in arcs:
			if arc.p == q:
				val = min(d[arc.n],(rprime + arc.w))
				if d[arc.n] != val:
					d[arc.n] = val
					p[arc.n] = p[arc.p].copy()
					p[arc.n].append(arc.t + ':' + arc.b)
					r[arc.n] = min(
						r[arc.n],
						(rprime + arc.w)
					)
					if arc.n not in queue:
						queue = [arc.n] + queue
	return d,p
