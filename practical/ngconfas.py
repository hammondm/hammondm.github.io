from ngrams import getngrams
import panphon as p
import random,re

wiki = 'fas4.txt'

#get letters and bigrams
lets,ngs = getngrams(wiki)

#fix g problem
newlets = set()
for let in lets:
	let = re.sub('g','ɡ',let)
	newlets.add(let)
lets = newlets

#get features
ft = p.FeatureTable()

fs = {}
for letter in lets:
	res = ft.word_fts(letter)
	if len(res) == 1:
		names = []
		names = res[0].names.copy()
		vals = res[0].items().copy()
		vals = {n:v for (n,v) in vals}
		names.append('trill')
		if letter == 'r':
			vals['trill'] = 1
		elif letter == 'ɾ':
			vals['trill'] = -1
		else:
			vals['trill'] = 0
		fs[letter] = p.segment.Segment(names,vals)
	else:
		fs[letter] = None

#get features for all bigrams with count of 0
nulls = []
for let1 in lets:
	for let2 in lets:
		key = (let1,let2)
		if key not in ngs:
			nulls.append((fs[let1],fs[let2]))

#function to look up segment from features
def lookup(feats):
	res = []
	if feats == None:
		res.append(None)
	else:
		for seg in fs:
			if fs[seg] != None and fs[seg] >= feats:
				res.append(seg)
	return res

print('initial constraints: ',len(nulls))

iters = 0
while True:
	iters += 1

	#LEFTSIDE REDUCTION

	#get right side features of constraints
	rightsides = []
	for (f,s) in nulls:
		if s not in rightsides:
			rightsides.append(s)

	#copy new constraints to this
	newones = []

	#go through the rightsides one by one
	for rightside in rightsides:

		#get all others with same right side
		candidates = []
		for (f,s) in nulls:
			if s == rightside:
				candidates.append((f,s))

		#randomize
		random.shuffle(candidates)

		#get the first candidate bits
		fst1 = candidates[0][0]
		snd1 = candidates[0][1]

		#skip if it's a word boundary
		if fst1 == None:
			for candidate in candidates:
				newones.append(candidate)
			continue

		#set of all possible left sides of
		#current right-set
		left = set()
		for candidate in candidates:
			if candidate[0] == None:
				left.add(None)
			else:
				ls = lookup(candidate[0])
				for l in ls:
					left.add(l)

		#find mergable constraints
		matches = []
		for candidate in candidates[1:]:
			if candidate[0] == None:
				continue
			#try the merge
			combo = fst1.intersection(candidate[0])
			#get all the segments that prediccts
			combomatches = set()
			for seg in fs:
				if fs[seg] != None and fs[seg] >= combo:
					combomatches.add(seg)
			canzero = lookup(candidate[0])
			if len(combomatches) > len(canzero) and \
					combomatches.issubset(left):
				matches.append(
					(lookup(candidate[0]),combomatches,combo)
				)
		#find the biggest match
		if len(matches) == 0:
			for candidate in candidates:
				newones.append(candidate)
		else:
			lengths = [len(x[1]) for x in matches]
			li = lengths.index(max(lengths))

			best = matches[li]
			bestcon = (best[2],snd1)

			#save new constraint
			newones.append(bestcon)
			for candidate in candidates[1:]:
				if candidate[0] == None:
					newones.append(candidate)
				elif not ((candidate[0] & bestcon[0]) \
						== bestcon[0]):
					newones.append(candidate)

	#RIGHTSIDE REDUCTION

	#get distinct right side features of constraints
	leftsides = []
	for (f,s) in newones:
		if f not in leftsides:
			leftsides.append(f)

	#copy new constraints to this
	realnewones = []

	#go through leftsides one by one
	for leftside in leftsides:

		#get all others with same right side
		candidates = []
		for (f,s) in newones:
			if f == leftside:
				candidates.append((f,s))

		#randomize
		random.shuffle(candidates)

		#get the first candidate bits
		fst1 = candidates[0][0]
		snd1 = candidates[0][1]

		#skip if it's a word boundary
		if snd1 == None:
			for candidate in candidates:
				realnewones.append(candidate)
			continue

		#set of all possible right sides of current left-set
		right = set()
		for candidate in candidates:
			if candidate[1] == None:
				right.add(None)
			else:
				rs = lookup(candidate[1])
				for r in rs:
					right.add(r)

		#find mergable constraints
		matches = []
		for candidate in candidates[1:]:
			if candidate[1] == None:
				continue
			#try the merge
			combo = snd1.intersection(candidate[1])
			#get all the segments that prediccts
			combomatches = set()
			for seg in fs:
				if fs[seg] != None and fs[seg] >= combo:
					combomatches.add(seg)
			canzero = lookup(candidate[1])
			if len(combomatches) > len(canzero) and \
					combomatches.issubset(right):
				matches.append(
					(lookup(candidate[1]),combomatches,combo)
				)
		#find the biggest match
		if len(matches) == 0:
			for candidate in candidates:
				realnewones.append(candidate)
		else:
			lengths = [len(x[1]) for x in matches]
			li = lengths.index(max(lengths))

			best = matches[li]
			bestcon = (fst1,best[2])

			#save new constraint
			realnewones.append(bestcon)
			for candidate in candidates[1:]:
				if candidate[1] == None:
					realnewones.append(candidate)
				elif not ((candidate[1] & bestcon[1]) \
						== bestcon[1]):
					realnewones.append(candidate)

	print('constraints: ',len(realnewones))

	#exit from while True:
	if len(nulls) == len(realnewones):
		break
	else:
		nulls = realnewones

for rno in realnewones:
	print(f'{rno[0]}:{rno[1]}')
	print(f'{lookup(rno[0])}:{lookup(rno[1])}\n')

print('iterations: ',iters)

