import os,librosa
from scipy.io import wavfile
from sklearn.cluster import KMeans
import numpy as np
import pomegranate as p
from pomegranate.hmm import DenseHMM
from pomegranate.distributions \
	import Categorical

where = '/Users/hammond/etexts/commands/'
segnodes = 2
maxiter = 10
order = 10
wlength = 130
clusters = 8
numtrain = 100

digits = [
	'zero','one','two','three','four',
	'five','six','seven','eight','nine'
]

#create stored digits
allscores = []
filelist = []
for digit in digits:
	digitset = []
	files = os.listdir(where+digit)
	filelist.append(files)
	for f in files:
		try:
			fs,w = wavfile.read(
				where + digit + '/' + f
			)
			w = w.astype(float)
			cur = 0
			res = []
			while cur+wlength <= len(w):
				lpc = librosa.lpc(
					w[cur:cur+wlength],
					order=order
				)
				res.append(lpc)
				cur += wlength
			res = np.array(res)
			digitset.append(res)
		except:
			print(f'error, skipping: {digit}/{f}')
		if len(digitset) == numtrain+10: break
	allscores.append(digitset)

#extract training items
train = []
for score in allscores:
	for digit in score[:numtrain]:
		train.append(digit)
train = np.vstack(train)

#use k-means to make clusters
print('clustering...')
km = KMeans(
	init='random',
	n_init='auto',
	n_clusters=clusters
)
km.fit(train)

#convert everything to VQ codes
allcodes = []
for digits in allscores:
	digitset = []
	for digit in digits:
		code = km.predict(digit)
		digitset.append(
			[[[c] for c in code]]
		)
	allcodes.append(digitset)

#make linear HMMs
print('creating HMMs...')
segments = np.array([4,3,2,3,3,4,4,5,3,4,3])
lengths = segments*segnodes + 2
clusterprob = 1/clusters
dist = {str(i):clusterprob for i \
	in range(clusters)}

hmms = []
for i in range(10):
	states = lengths[i]
	m = DenseHMM(
		max_iter=maxiter,
		verbose=True,
		inertia=0.2
	)
	#states
	statelist = []
	for s in range(states):
		s = Categorical([list(dist.values())])
		statelist.append(s)
	m.add_distributions(statelist)
	#start prob
	m.add_edge(m.start,statelist[0],1.0)
	#final state
	m.add_edge(statelist[-1],m.end,0.5)
	#loop transitions
	for state in statelist:
		m.add_edge(state,state,0.5)
	#sequential transitions
	for i in range(len(statelist)-1):
		m.add_edge(
			statelist[i],
			statelist[i+1],
			0.5
		)
	hmms.append(m)

#train HMMs
print('training...')
for i in range(10):
	print(i)
	trainset = allcodes[i][:numtrain]
	hmm = hmms[i]
	hmm.fit(trainset)

#test HMMs
print('testing...')
total = 0
for i in range(10):
	testset = allcodes[i][numtrain:]
	for testitem in testset:
		allres = []
		for hmm in hmms:
			res = hmm.probability(testitem)
			itm = res.item()
			allres.append(itm)
		allres = np.array(allres)
		if all(np.isnan(allres)):
			idx = 50
		else:
			idx = np.nanargmax(allres)
		if idx == i: total += 1

print(f'Correct: {total}/100')

