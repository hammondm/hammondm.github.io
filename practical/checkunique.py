import panphon as p

#f = open('fasletters.txt','r')
f = open('faslet2.txt','r')
t = f.read()
f.close()

lets = t.split()

ft = p.FeatureTable()

fs = {}
for letter in lets:
	res = ft.word_fts(letter)
	if len(res) == 1:
		fs[letter] = res[0]

def lookup(fsdic,sound):
	res = []
	for s in fsdic:
		if fsdic[s] == sound:
			res.append(s)
	return res

for s1 in fs:
	s1f = fs[s1]
	res = []
	for s2 in fs:
		if s1f == fs[s2]:
			res.append(s2)
	print(s1,res)

