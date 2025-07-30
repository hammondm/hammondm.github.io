import os
from scipy.io import wavfile
import numpy as np

digits = [
	'zero','one','two','three','four',
	'five','six','seven','eight','nine'
]

where = '/Users/hammond/etexts/commands/'

#initialize results
allscores = []
testscores = []
#go through digits one by one
for digit in digits:
	files = os.listdir(where+digit)
	scores = []
	#go through 10 examples of each
	for f in files[:10]:
		fs,w = wavfile.read(
			where + digit + '/' + f
		)
		#calculate mean
		score = np.mean(np.abs(w))
		scores.append(score)
	allscores.append(np.mean(scores))
	scores = []
	#go through the next 10
	for f in files[10:20]:
		fs,w = wavfile.read(
			where + digit + '/' + f
		)
		#calculate means again
		score = np.mean(np.abs(w))
		scores.append(score)
	scores = np.array(scores)
	testscores.append(scores)

allscores = np.array(allscores)

#get matches
matches = 0
for i in range(len(testscores)):
	testset = testscores[i]
	for item in testset:
		res = np.argmin(np.abs(item - allscores))
		if res == i: matches += 1

#print results
print(matches)
