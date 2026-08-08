import panphon as p

f = open('letters.txt','r')
t = f.read()
f.close()

t = t.split('\n')
t = t[:-1]

ft = p.FeatureTable()

fs = {}
for v in t:
	fs[v] = ft.word_fts(v)[0]

for v in fs:
	if fs[v]['syl'] == 1 and fs[v]['cons'] == -1:
		print(v,end=' ')
print()

