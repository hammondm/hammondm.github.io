#collect minimal pairs of sounds from wikipron output

#filename = 'wikiproncym.txt'
filename = 'wikipronfas.txt'

#read in data
f = open(filename,'r')
t = f.read()
f.close()

#break into lines
t = t.split('\n')[:-1]

#collect word-transcription pairs sorted by length
lengths = {}
for line in t:
	word,trans = line.split('\t')
	trans = trans.split(' ')
	tlen = len(trans)
	if tlen in lengths:
		lengths[tlen].append((word,trans))
	else:
		lengths[tlen] = [(word,trans)]

#go through each length set
pairs = set()
for length in lengths:
	x = lengths[length]
	#get the items one by one
	while len(x) > 0:
		w1,t1 = x.pop()
		#compare to all remaining items
		for w2,t2 in x:
			#check that they're spelled differently
			if w1 != w2:
				#how many transcription differences
				diffs = [(i,j) for i,j in zip(t1,t2) if i != j]
				#if it's a minimal pair, add sound pair to list
				if len(diffs) == 1:
					pairs.add(tuple(sorted(diffs[0])))

#print out results
for s1,s2 in sorted(pairs):
	print(s1,s2)

