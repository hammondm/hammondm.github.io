import panphon.sonority as ps

filename = 'wikicym3.txt'

s = ps.Sonority()

#read in data
f = open(filename,'r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#go through the data word by word
for line in lines:
	left,right = line.split('\t')
	letters = right.split()
	#get sonority for all segments
	numbers = [s.sonority(letter) for letter in letters]
	#syllabify if there's more than one segment
	if len(numbers) > 1:
		i = 1
		m = len(numbers) - 1
		breaks = []
		while i < m:
			if numbers[i] <= numbers[i-1] and \
				numbers[i] < numbers[i+1]:
					breaks.append(i)
			i += 1
		breaks.reverse()
		for b in breaks:
			letters.insert(b,'$')
	print(f'{left}: {''.join(letters)}')

