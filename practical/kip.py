import re

#generalizations about progressive <Vp>

vowels = "aeiouAEIOU"

f = open('kortho.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

counts = {}
for line in lines:
	if re.search('V;PROG;SG;1;PRS',line):
		lemma,form,tags = line.split('\t')
		root = lemma.split(' ')[-1]
		sfxvs = root[-2:]
		ipform = form.split(' ')[-2][-2:]
		stem = root[:-2]
		stemvs = [v for v in stem if v in vowels]
		if len(stemvs) == 0:
			term = 'X'
		else:
			term = stemvs[-1]

			term = sfxvs + '-' + stemvs[-1]

		print(f'{line}:\n\t{stemvs}')
		if term in counts:
			if ipform in counts[term]:
				counts[term][ipform] += 1
			else:
				counts[term][ipform] = 1
		else:
			counts[term] = {ipform:1}

for rv in sorted(counts):
	print(rv)
	for ip in counts[rv]:
		print(f'\t{ip}: {counts[rv][ip]}')

