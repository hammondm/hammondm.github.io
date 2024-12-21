import os,re,librosa,sklearn.mixture
import numpy as np

#location of speech commands data
#dirname = '/data/commands/'
dirname = '/Users/hammond/etexts/commands/'

#number of speakers
speaknum = 20

#number of training items per speaker
speakmax = 30

#number of test items per speaker
testcount = 10

#get all file names
files = os.listdir(dirname)

#speaker list with counts
speakers = {}

#go through the file names
for f in files:
	#look for subdirectories
	if os.path.isdir(dirname+f):
		if f[0] != '_':
			#get names of wav files
			subfiles = os.listdir(dirname+f)
			#extract and save speaker IDs
			for fx in subfiles:
				fx2 = re.sub('_nohash.*','',fx)
				if fx2 in speakers:
					speakers[fx2] += 1
				else:
					speakers[fx2] = 1

#sort by counts
itms = sorted(
	list(speakers.items()),
	key=lambda x: x[1],
	reverse=True
)

#5 speakers with the most data
nbest = []
ngmm = {}
ndata = {}
for itm in itms[:speaknum]:
	nbest.append(itm[0])
	print(f'speaker {itm[0]}: {itm[1]}')
	gmm = sklearn.mixture.GaussianMixture(
		n_components=4,
		covariance_type='diag'
	)
	ngmm[itm[0]] = gmm
	ndata[itm[0]] = []

#go through the file names again
for f in files:
	#look for subdirectories
	if os.path.isdir(dirname+f):
		if f[0] != '_':
			#get names of wav files
			subfiles = os.listdir(dirname+f)
			for fx in subfiles:
				#get speaker id
				speakid = re.sub('_nohash.*','',fx)
				if speakid in nbest:
					#get MFCCs
					s,r = librosa.load(dirname+f+'/'+fx)
					mfcc = librosa.feature.mfcc(
						y=s,
						sr=r,
						n_mfcc=10,
						#25msec windows
						win_length=((r//1000) * 25),
						#10msec hops
						hop_length=((r//1000) * 10)
					)
					ndata[speakid].append(mfcc)

#split into train and test
tests = {}
for speakid in ndata:
	alldata = np.concatenate(
		ndata[speakid]
			[testcount:(speakmax+testcount)],
		axis=1
	)
	tests[speakid] = ndata[speakid]\
		[:testcount]
	#train GMMs on training data
	ngmm[speakid].fit(alldata.T)

#test
count = 0
good = 0
for speakid in tests:
	#get each training item
	for itm in tests[speakid]:
		count += 1
		results = {}
		#compare against each GMM
		for otherid in ngmm:
			res = ngmm[otherid].score(itm.T)
			results[otherid] = res.mean()
		#choose best score
		best = max(results,key=results.get)
		if best == speakid:
			good += 1

print(f'results: {good}/{count}: ',end='')
print(f'{good/count}')

