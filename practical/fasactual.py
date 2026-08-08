wiki = 'wikipronfas.txt'

f = open(wiki,'r')
t = f.read()
f.close()

t = t.split('\n')
t = t[:-1]

ones = set()
twos = set()
threes = set()
fours = set()

for line in t:
	word,trans = line.split('\t')
	sounds = trans.split(' ')
	if len(sounds) == 1:
		ones.add(trans)
	elif len(sounds) == 2:
		twos.add(trans)
	elif len(sounds) == 3:
		threes.add(trans)
	elif len(sounds) == 4:
		fours.add(trans)

print(f'ones: {len(ones)}')
print(f'twos: {len(twos)}')
print(f'threes: {len(threes)}')
print(f'fours: {len(fours)}')

