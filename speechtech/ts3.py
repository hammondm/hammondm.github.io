from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy

plt.rcParams['savefig.dpi'] = 300

sr,w = wavfile.read('mha.wav')
w = w.astype(float)

order = 10
rows = order + 1
columns = len(w)//1000
res = np.zeros([rows,columns])

#make LPC
for i in range(columns):
	start = i * 1000
	lpc = librosa.lpc(
		w[start:start+1000],
		order=order
	)
	res[:,i] = lpc

#make a new wave
newwave = np.zeros(columns*1000)
for i in range(columns):
	start = i * 1000
	f1 = scipy.signal.lfilter(
		[1.],
		res[:,i],
		w[start:start+1000]
	)
	newwave[start:start+1000] = f1

#average amplitude of old wave
wmean = np.mean(np.abs(w))
#average amplitude of the new wave
newwavemean = np.mean(np.abs(newwave))
#ratio
scale = newwavemean/wmean

#get error
e = w[:48000]*scale - newwave

#plot old wave
plt.subplot(2,1,1)
plt.title('old and new waves')
plt.plot(w[:5000]*scale)
#plot new wave
plt.subplot(2,1,2)
plt.plot(newwave[:5000])
plt.show()

#plot error
plt.plot(e[:1000])
plt.title('error')
plt.show()

#make wave from LPC and error/residual:
reswave = np.zeros(columns*1000)
for i in range(columns):
	start = i * 1000
	end = start + 1000
	f2 = scipy.signal.lfilter(
		[1.],
		res[:,i],
		e[start:end]
	)
	reswave[start:end] = f2

#plot new wave
plt.plot(reswave[:1000])
plt.title('using residual')
plt.show()

#raise or lower from 1000 to chage pitch
change = 1100

newreswave = np.zeros(columns*change)

for i in range(columns):
	estart = i * 1000
	eend = estart + 1000
	down = scipy.signal.resample(
		e[estart:estart+change],
		1000
	)
	f2 = scipy.signal.lfilter(
		[1.],
		res[:,i],
		down
	)
	newreswave[estart:eend] = f2

#plot new wave
plt.plot(newreswave[:1000])
plt.title('changing pitch')
plt.show()

