import re

filename = '/Users/hammond/etextsSMALL/cv-corpus-27.0-2026-09-11/ky/validated.tsv'

f = open(filename,'r')
t = f.read()
f.close()

t = t.split('\n')
t = t[1:-1]

words = set()

letters = set()

for line in t:
	fields = line.split('\t')
	s = fields[3].lower()
	s = re.sub('[\–”‘\-\—\!¬“,\'"\.\?;:]',' ',s)
	for letter in s:
		letters.add(letter)
	s = s.strip()
	s = re.split(' +',s)
	for word in s:
		words.add(word)

for word in words:
	print(word)

