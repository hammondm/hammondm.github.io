import re

#generalizations about progressive <Vp>

#do stems and lemma suffixes too

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
			finalv = 'X'
		else:
			finalv = stemvs[-1]
			#finalv = stemvs[0]
		print(f'{line}:\n\t{stemvs}')
		if finalv in counts:
			if ipform in counts[finalv]:
				counts[finalv][ipform] += 1
			else:
				counts[finalv][ipform] = 1
		else:
			counts[finalv] = {ipform:1}

for rv in counts:
	print(rv)
	for ip in counts[rv]:
		print(f'\t{ip}: {counts[rv][ip]}')

