from ngrams import getngrams
import panphon as p
import re

wiki = 'estfixed.tsv'

#get letters
lets,_ = getngrams(wiki)

#remove word boundary
lets.remove('#')

ft = p.FeatureTable()
fs = {}
for letter in lets:
   res = ft.word_fts(letter)
   if len(res) == 1:
      itms = res[0].items().copy()
      feats = [n for (n,_) in itms]
      vals = {n:v for (n,v) in itms}
      feats.append('halflong')
      if re.search('ˑ',letter):
         vals['halflong'] = 1
      elif re.search('ːː',letter):
         vals['halflong'] = 1
      else:
         vals['halflong'] = -1
      if re.search('ː',letter):
         vals['long'] = 1
      if re.search('d̥ʲ',letter):
         vals['hi'] = 1
      fs[letter] = p.segment.Segment(feats,vals)
   else:
      fs[letter] = None

#look up segments from features
def lookup(s):
	res = []
	for x in fs:
		if fs[x] >= s:
			res.append(x)
	return res

#get all feature names
fnames = [i[0] for i in fs['ɑ'].items()]

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

