import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,rfftfreq
from scipy.io import wavfile

fs,w = wavfile.read('mha.wav')

#spectrum
spec = rfft(w)
freqs = rfftfreq(len(w),1/fs)
#log scale
powerspec = np.abs(spec)**2
#cepstrum
ceps = irfft(np.log(powerspec))

#plot
plt.subplot(3,1,1)
plt.plot(w)
plt.subplot(3,1,2)
plt.plot(freqs[:2000],powerspec[:2000])
plt.subplot(3,1,3)
#noise in lowest cepstrum values
plt.plot(np.abs(ceps)[10:500])
plt.show()
