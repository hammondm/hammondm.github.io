import shutil

#original wav location
fromloc = '/data/' + \
	'cv-corpus-16.0-2023-12-06/cy/clips/'
#location for copied wav files
toloc = '/data/mhcy/wavs/'
#metadata for the one speaker
f = open('onespeaker.csv','r')
t = f.read()
f.close()
t = t.split('\n')
t = t[:-1]

#copy wav files
for line in t:
	bits = line.split('|')
	filename = bits[0]
	shutil.copy(fromloc+filename+'.wav',toloc)
