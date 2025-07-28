import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,rfftfreq
from scipy.io import wavfile
from scipy.fftpack import dct

fs,w = wavfile.read('mha.wav')

#spectrum
spec = rfft(w)
freqs = rfftfreq(len(w),1/fs)
#log scale
powerspec = np.abs(spec)**2

#discrete cosine transform
d = dct(w)

plt.subplots_adjust(hspace=0.5)

#plot
plt.subplot(3,1,1)
plt.plot(w)
plt.subplot(3,1,2)
plt.plot(
	freqs[:2000],
	powerspec[:2000]
)
plt.subplot(3,1,3)
plt.plot(np.abs(d[:4000]))
plt.show()

