#metadata file location
val = '/data/cv-corpus-16.0-2023-12-06/' + \
	'cy/validated.tsv'

#read metadata
f = open(val,'r')
t = f.read()
f.close()
#split into lines
t = t.split('\n')
t = t[1:]
t = t[:-1]

#get counts for each speaker
speakers = {}
for line in t:
	bits = line.split('\t')
	if len(bits) > 0:
		speaker = bits[0]
		if speaker in speakers:
			speakers[speaker] += 1
		else:
			speakers[speaker] = 1

#sort counts
speakerItemsSorted = sorted(
	speakers.items(),
	key=lambda x: x[1]
)

#get best speaker
best = speakerItemsSorted[-1][0]

#make new metadata file from that speaker
g = open('onespeaker.csv','w')
for line in t:
	bits = line.split('\t')
	if len(bits) > 2 and bits[0] == best:
		g.write(
			f'{bits[1][:-4]}|{bits[2]}|{bits[2]}\n'
		)
g.close()
