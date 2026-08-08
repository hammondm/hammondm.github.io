import panphon.sonority as ps
import matplotlib.pyplot as plt
import sys

#display sonority profile of a word

wiki = 'wikicym3.txt'

if len(sys.argv) != 2:
	print('usage: python sonshow.py word')
	exit()

word = sys.argv[1]

print(f'your word: {word}')

f = open(wiki,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

trans = None
for line in lines:
	w,p = line.split('\t')
	if w == word:
		trans = p.split()
		break

if not trans:
	print('Your word is not in the data set')
	exit()

print(trans)

s = ps.Sonority()

numbers = []
for x in trans:
	numbers.append(s.sonority(x))

print(numbers)

plt.plot(numbers)
plt.xticks(range(len(trans)),labels=trans)
plt.show()

