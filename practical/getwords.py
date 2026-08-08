import re

filename = 'cv-corpus-19.0-2024-09-13/cy/validated.tsv'

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
