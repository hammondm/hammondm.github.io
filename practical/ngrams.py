def getngrams(filename):
	#open file
	f = open(filename,'r')
	t = f.read()
	f.close()

	#break into lines
	t = t.split('\n')
	t = t[:-1]

	#initialize results
	letters = set()
	ngrams = {}

	#go through all the lines
	for line in t:
		bits = line.split('\t')
		sounds = bits[1].split(' ')
		sounds = ['#'] + sounds + ['#']
		i = 0
		#go through all the sounds
		while i < len(sounds)-1:
			letters.add(sounds[i])
			key = (sounds[i],sounds[i+1])
			if key in ngrams:
				ngrams[key] += 1
			else:
				ngrams[key] = 1
			i += 1
	return letters,ngrams

if __name__ == "__main__":

	#wiki = 'wikiproncym.txt'
	wiki = 'wikipronfas.txt'

	lets,ngs = getngrams(wiki)
	#print results
	for let1 in lets:
		for let2 in lets:
			key = (let1,let2)
			if key in ngs:
				print(f'{let1}, {let2}: {ngs[key]}')
			else:
				print(f'{let1}, {let2}: 0')

