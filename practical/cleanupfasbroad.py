import re

filename = 'fas_arab_broad.tsv'

f = open(filename,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

for line in lines:
	#remove suffixes
	if '‿' in line: continue
	#replace weird g
	line = re.sub('ɡ','g',line)
	#get rid of palatalized k
	line = re.sub('kʲ','k',line)
	#replace incorrect long a
	line = re.sub('â','aː',line)
	#replace incorrect umlaut a
	line = re.sub('ä','a',line)
	#remove errant stress mark
	line = re.sub('ɒ́ː','ɒː',line)
	#split alternate lines
	if '~' in line:
		leftside,rightside = line.split('\t')
		first,second = rightside.split(' ~ ')
		print(leftside + '\t' + first)
		print(leftside + '\t' + second)
	else:
		print(line)

