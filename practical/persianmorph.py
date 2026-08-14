import re

#check persian morphology

#filename = 'uni/fas'
filename = 'wikifasortho.txt'

f = open(filename,'r')
t = f.read()
f.close()

#lines = t.split('\n')[1:-1]
lines = t.split('\n')[:-1]

print('number of forms:',len(lines))

verbs = {}
for line in lines:
	lemma,form,tags = line.split('\t')
	if lemma not in verbs:
		verbs[lemma] = set()
	verbs[lemma].add((form,tags))

print('number of lemmas:',len(verbs))

velem = set()
complexpreds = 0
for lemma in verbs:
	bits = lemma.split()
	if len(bits) > 1:
		complexpreds += 1
		velem.add(bits[-1])

print('complex predicates:',complexpreds)

print('unique verbal elements:',len(velem))

hapax = 0
for ve in velem:
	if ve not in verbs:
		hapax += 1

print("verbal elements that don't occur alone:",hapax)

