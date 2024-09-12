import os,librosa
from scipy.io import wavfile
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from pomegranate.gmm import \
	GeneralMixtureModel
import numpy as np
import pomegranate as p
import torch as t

#use mixture model
mix = False
#how many states
states = 3
#order of the lpc
order = 3
#size of lpc window
wlength = 130
#training items per digit
numtrain = 50
#training iterations
maxiter = 10
#test items per digit
testitems = 10

digits = [
	'zero','one','two','three','four',
	'five','six','seven','eight','nine'
]

where = '/mhdata/commands/'

#create stored digits
allscores = []
filelist = []
#go digit by digit
for digit in digits:
	digitset = []
	files = os.listdir(where+digit)
	filelist.append(files)
	#go through each instance/file
	for f in files:
		try:
			fs,w = wavfile.read(
				where + digit + '/' + f
			)
			w = w.astype(float)
			cur = 0
			res = []
			#lpc for each window
			while cur+wlength <= len(w):
				lpc = librosa.lpc(
					w[cur:cur+wlength],
					order=order
				)
				res.append(lpc[1:])
				cur += wlength
			resprime = t.tensor(np.array([res]))
			digitset.append(resprime)
		except:
			print(f'error, skipping: {digit}/{f}')
		if len(digitset) == numtrain+testitems: \
			break
	allscores.append(digitset)

#make HMMs
print('creating HMMs...')
hmms = []
for i in range(10):
	print(f'making HMM #{i}')
	m = DenseHMM(
		verbose=True,
		max_iter=maxiter
	)
	dists = []
	#add states
	for i in range(states):
		#mixture model
		if mix:
			dists.append(
				GeneralMixtureModel(
					[Normal(),Normal()]
				)
			)
		#simple multivariate
		else:
			dists.append(Normal())
	m.add_distributions(dists)
	hmms.append(m)

#train HMMs
print('training...')
for i in range(10):
	print(i)
	trainset = allscores[i][:numtrain]
	hmm = hmms[i]
	hmm.fit(trainset)

#test HMMs
print('testing...')
total = 0
for i in range(10):
	testset = \
		allscores[i][numtrain:numtrain+testitems]
	for testitem in testset:
		allres = []
		for hmm in hmms:
			res = hmm.probability(testitem)
			itm = res.item()
			allres.append(itm)
		#handle NaNs
		allres = np.array(allres)
		if all(np.isnan(allres)):
			idx = 50
		else:
			idx = np.nanargmax(allres)
		if idx == i: total += 1

totalitems = 10*testitems
print(f'Correct: {total}/{totalitems}')

