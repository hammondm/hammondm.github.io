import os,librosa,random
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#location of speech commands data
commands = '/data/commands/'
#size of window
winsize = 3000
#vad threshold
threshold = 200
#mfcc coefficients
coefs = 10

#digit names
digits = 'zero one two three four ' + \
	'five six seven eight nine'
digits = digits.split(' ')

#collect speakers for each digit
allspeakers = []
fulllist = set()
for i,digit in enumerate(digits):
	speakers = set()
	files = os.listdir(commands + digit)
	for filename in files:
		speaker = filename.split('_')[0]
		speakers.add(speaker)
		fulllist.add(speaker)
	allspeakers.append(speakers)

#speakers who have all digits
overlaplist = set()
for speaker in fulllist:
	res = map(
		lambda x: speaker in x,
		allspeakers
	)
	if all(res): overlaplist.add(speaker)
print(f'{len(overlaplist)} speakers')

#choose two random speakers
one,two = random.sample(overlaplist,2)
print('two random speakers:',one,two)
#a minimum of three items for each speakers
items = [one,one,one,two,two,two]
#add 4 additional randomly chosen items
for i in range(4):
	items.append(random.choice([one,two]))
#randomize the list
random.shuffle(items)
#print the speakers
print(items)

#concatenate digits
wall = []
for i,s in enumerate(items):
	digit = digits[i]
	filename = s + '_nohash_0.wav'
	fs,w = wavfile.read(
		commands + digit + '/' + filename
	)
	wall.append(w)
wall = np.concatenate(wall)

#do VAD
ms = []
thresh = []
i = winsize
last = 0
spans = []
while i < len(wall):
	vals = wall[i-winsize:i]
	i += winsize
	m = np.abs(vals).mean()
	ms.append(m)
	if m > threshold:
		if last == 0:
			start = i - winsize
		last = 1
	else:
		if last == 1:
			end = i
			spans.append((start,end))
		last = 0
	thresh.append(last)

#do MFCCs
mfccs = []
for span in spans:
	sp = wall[span[0]:span[1]]
	mfcc = librosa.feature.mfcc(
		y=sp.astype(float),
		sr=fs,
		n_mfcc=coefs,
		#25msec windows
		win_length=((fs//1000) * 25),
		#10msec hops
		hop_length=((fs//1000) * 10)
	)
	mfccs.append(np.mean(mfcc,axis=1))

#use k-means to make clusters
print('clustering...')
km = KMeans(
        init='random',
        n_clusters=2
)
km.fit(mfccs)

for mfcc in mfccs:
	print(km.predict(
		np.expand_dims(mfcc,axis=0)
	))

#plot results
plt.subplot(3,1,1)
plt.plot(wall)
plt.subplot(3,1,2)
plt.plot(ms)
plt.subplot(3,1,3)
plt.plot(thresh)
plt.show()
