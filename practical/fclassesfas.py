from ngrams import getngrams
import panphon as p
import re

wiki = 'fas4.txt'

#get letters
lets,_ = getngrams(wiki)

#remove word boundary
lets.remove('#')

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

#look up segments from features
def lookup(s):
	res = []
	for x in fs:
		if fs[x] >= s:
			res.append(x)
	return res

#get all feature names
fnames = [i[0] for i in fs['a'].items()]

#all possible feature values
vals = [(1,'plus'),(0,'unmarked'),(-1,'minus')]

#import statement for generated classes
print("from pyfoma import FST\n")

#go through all the features
for fname in fnames:
	#go through all possible values
	for (val,vname) in vals:
		#get the segment class
		res = lookup({fname:val})
		#if it has members, make the class
		if len(res) > 0:
			con = '"(' + '|'.join(res) + ')"'
			print(f'{fname}{vname} = FST.re({con})\n')

