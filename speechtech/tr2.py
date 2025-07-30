import os,librosa
from scipy.io import wavfile
import numpy as np

order = 4

digits = [
	'zero','one','two','three','four',
	'five','six','seven','eight','nine'
]

where = '/Users/hammond/etexts/commands/'

#create stored digits
allscores = []
filelist = []
for digit in digits:
	files = os.listdir(where+digit)
	filelist.append(files)
	fs,w = wavfile.read(
		where + digit + '/' + files[0]
	)
	wlength = len(w)//10
	w = w.astype(float)
	cur = 0
	res = []
	#make LPCs for windows
	while cur+wlength <= len(w):
		lpc = librosa.lpc(
			w[cur:cur+wlength],
			order=order
		)
		cur += wlength
		res.append(lpc)
	res = np.array(res)
	allscores.append(res)

#test 100 digits
matches = 0
for i in range(len(filelist)):
	digit = digits[i]
	files = filelist[i]
	for f in files[1:11]:
		fs,w = wavfile.read(
			where + digit + '/' + f
		)
		wlength = len(w)//10
		w = w.astype(float)
		cur = 0
		res = []
		#make LPCs
		while cur+wlength <= len(w):
			lpc = librosa.lpc(
				w[cur:cur+wlength],
				order=order
			)
			cur += wlength
			res.append(lpc)
		res = np.array(res)
		scores = []
		#compare with stored digits
		for score in allscores:
			val = np.sqrt(np.sum((res - score)**2))
			scores.append(val)
		if np.argmin(scores) == i: matches += 1

#print results
print(matches)
