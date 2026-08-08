from pyfoma import FST

#read in paradigms
f = open('1sg.txt','r')
t = f.read()
f.close()
t = t.split('\n')[:-1]

#create tag string
tags = FST.re("'V' ; 'COL' ; 1 ; 'SG' ; 'IND' ; 'PRS'")
tagstr = "V;COL;1;SG;IND;PRS"
tagdict = {'tags':tags}

#create separate FSTs for each verb
fstlist = []
for line in t:
	bits = line.split('\t')
	outbit = bits[1].split(' ')
	s = "$^rewrite((" + bits[0] + \
		" ; $tags):(" + outbit[0] + \
		" ' ' " + outbit[1] + "))"
	fstlist.append(FST.re(s,tagdict))

print(f'specific rules: {len(fstlist)}')

#compose FSTs
everything = fstlist[0]
for fst in fstlist[1:]:
	everything = everything.compose(fst)

#minimize FST
everything = everything.minimize()

#get arcs and states
print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

#test
print(list(everything.generate('addo;' + tagstr)))
print(list(everything.generate('mynd;' + tagstr)))

