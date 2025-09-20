import numpy as np
import matplotlib.pyplot as plt
import librosa
from sp22 import lpc2formants
import sp21

plt.rcParams['savefig.dpi'] = 300

def getformants(wave,rate):
	i = 0
	#samples per window
	span = 1102
	win = 1
	x = []
	wave = wave.astype(float)
	#go window by window
	while i < len(wave):
		i += span
		win += 1
		if i + span < len(wave):
			#10th order lpc
			x.append(sp21.lpc(wave[i:i+span],10))
	x = np.array(x)
	formants = []
	size = x.shape[0]
	for i in range(size):
		res = lpc2formants(x[i],rate)
		formants.append(res[0])
	return formants

#resample to focus on lower frequencies
w,fs = librosa.load('mha.wav',sr=8000)
aformants = getformants(w,fs)
w,fs = librosa.load('mhi.wav',sr=8000)
iformants = getformants(w,fs)

plt.subplot(1,2,1)
plt.plot(aformants,'b')
plt.subplot(1,2,2)
plt.plot(iformants,'b')
plt.show()
