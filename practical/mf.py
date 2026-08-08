filename = 'uni/cym'

f = open(filename,'r')
t = f.read()
f.close()

t = t.split('\n')
t = t[1:-1]

fs = set()
combos = set()

for line in t:
	lemma,word,features = line.split('\t')
	combos.add(features)
	features = features.split(';')
	for feature in features:
		fs.add(feature)

print(len(fs))
for f in sorted(fs):
	print(f)

print(len(combos))
for combo in sorted(combos):
	print(combo)

