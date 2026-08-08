#filename = 'wikiproncym.txt'
#filename = 'wikipronfas.txt'
#filename = 'fas_arab_broad.tsv'
filename = 'fas4.txt'

f = open(filename,'r')
t = f.read()
f.close()

t = t.split('\n')
t = t[:-1]

d = {}

for line in t:
	_,sounds = line.split('\t')
	ss = sounds.split(' ')
	for s in ss:
		if s in d:
			d[s] += 1
		else:
			d[s] = 1

for s in d:
	print(f'{s}: {d[s]}')
