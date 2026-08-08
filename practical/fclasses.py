from ngrams import getngrams
import panphon as p

wiki = 'wikicym3.txt'

#get letters
lets,_ = getngrams(wiki)

#remove word boundary
lets.remove('#')

#get features
ft = p.FeatureTable()
fs = {}
for letter in lets:
	res = ft.word_fts(letter)
	fs[letter] = res[0]

#fix [ŋ̊]
fs['ŋ̊'] = p.segment.Segment(['syl','son','cons','cont',
			'delrel','lat','nas','strid','voi','sg','cg',
			'ant','cor','distr','lab','hi','lo','back',
			'round','velaric','tense','long','hitone',
			'hireg'],
			{'syl': -1,'son': 1,'cons': 1,'cont': -1,
			'delrel': -1,'lat': -1,'nas': 1,'strid': -1,
			'voi': -1,'sg': -1,'cg': -1,'ant': -1,'cor': -1,
			'distr': 0,'lab': -1,'hi': 1,'lo': -1,'back': 1,
			'round': -1,'velaric': -1,'tense': 0,'long': -1,
			'hitone': 0,'hireg': 0})

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

