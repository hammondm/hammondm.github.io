#location of files
dirname = 'UD_Welsh-CCG/'
#individual file names
filenames = [
	'cy_ccg-ud-dev.conllu',
	'cy_ccg-ud-train.conllu',
	'cy_ccg-ud-test.conllu'
]

#go through all the files
dataset = set()
for filename in filenames:
	print(dirname+filename)
	f = open(dirname+filename,'r')
	t = f.read()
	t = t.split('\n')
	for line in t:
		#ignore comments
		if '#' not in line:
			#break into fields
			bits = line.split('\t')
			#ignore bad lines
			if len(bits) > 5:
				dataset.add(tuple(bits[1:6]))
	f.close()

#print number of distinct items
#print(f'set: {len(dataset)}')

#organize by lemma
data = {}
for item in dataset:
	key = item[1]
	if key in data:
		data[key].append(item)
	else:
		data[key] = [item]

#print out sorted by lemma
for key in sorted(data.keys()):
	vals = data[key]
	for val in vals:
		print(f'{key}\t{val[0]}\t{val[2]}\t{val[3]}\t{val[4]}')

