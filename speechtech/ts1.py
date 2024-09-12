import librosa
from scipy.io import wavfile
import numpy as np
import scipy

def buzz(lpc,freq,sr):
	#make buzz
	b = scipy.signal.sawtooth(
		2*np.pi*freq*np.arange(10000)/sr
	)
	#add it to lpc
	res = scipy.signal.lfilter([1.],lpc,b)
	return res

#try it with [i]
sr,w = wavfile.read('mhi.wav')
lpc = librosa.lpc(
	w.astype(float),
	order=10
)
res = buzz(lpc,200,sr)
wavfile.write('ires.wav',sr,res)

#try it with [a]
sr,w = wavfile.read('mha.wav')
lpc = librosa.lpc(
	w.astype(float),
	order=10
)
res = buzz(lpc,200,sr)
wavfile.write('ares.wav',sr,res)
