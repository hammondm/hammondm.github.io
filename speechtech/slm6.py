from slm5 import Arc

def parsefst(f):
	#break into lines
	fst = f.split('\n')
	arcs = []
	finalstates = {}
	#go through the lines
	for line in fst:
		bits = line.split('\t')
		#handle arcs
		if len(bits) == 5:
			a = Arc(*bits)
			arcs.append(a)
		#handle final states
		elif len(line) > 0:
			finalstates[bits[0]] = float(bits[1])
	return arcs,finalstates
