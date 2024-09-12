import multiprocessing as mp
import os,librosa
import numpy as np
from scipy.io import wavfile
from tr4 import sssd,dtwmin

order = 9
wlength = 120
processes = 10

digits = [
	'zero','one','two','three','four',
	'five','six','seven','eight','nine'
]

where = '/data/commands/'

#create stored digits
allscores = []
filelist = []
for digit in digits:
	files = os.listdir(where+digit)
	filelist.append(files)
	fs,w = wavfile.read(
		where + digit + '/' + files[0]
	)
	w = w.astype(float)
	cur = 0
	res = []
	#go through windows making LPCs
	while cur+wlength <= len(w):
		lpc = librosa.lpc(
			w[cur:cur+wlength],
			order=order
		)
		cur += wlength
		res.append(lpc)
	res = np.array(res)
	allscores.append(res)

#separate test function
def test(i):
	matches = 0
	digit = digits[i]
	files = filelist[i]
	for f in files[1:11]:
		fs,w = wavfile.read(
			where + digit + '/' + f
		)
		w = w.astype(float)
		cur = 0
		res = []
		#go through windows, making LPCs
		while cur+wlength <= len(w):
			lpc = librosa.lpc(
				w[cur:cur+wlength],
				order=order
			)
			cur += wlength
			res.append(lpc)
		res = np.array(res)
		scores = []
		#do DTW
		for score in allscores:
			val = dtwmin(res.T,score.T)
			scores.append(val)
		#get best score
		if np.argmin(scores) == i: matches += 1
	return matches

#start multiprocessing
mypool = mp.Pool(processes=processes)
#map function to all digits
allmatches = mypool.map(
	test,
	range(len(filelist))
)

print(sum(allmatches))
