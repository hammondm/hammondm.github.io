#collect stats on segment distribution in newdic

import matplotlib.pyplot as plt

newdic = 'newdic'

f = open(newdic,'r')
t = f.read()
f.close()

t = t.split('\n')
t = t[:-1]

letters = {}

for line in t:
	bits = line.split('\t')
	trans = bits[0]
	for letter in trans:
		if letter in letters:
			letters[letter] += 1
		else:
			letters[letter] = 1

vals = sorted(
	letters.items(),
	key=lambda x: x[1]
)

for val in vals:
	print(val)

nums = [val[1] for val in vals]

plt.plot(nums)
plt.show()

