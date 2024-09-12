import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft, \
	rfftfreq,fft,ifft
from scipy.io import wavfile

fs,w = wavfile.read('mha.wav')

#spectrum
spec = rfft(w)
freqs = rfftfreq(len(w),1/fs)
#remove imaginary parts
pspec = np.abs(spec)

#inverse real fourier
recon = irfft(spec)

#plot
plt.subplot(3,1,1)
plt.plot(w)
plt.subplot(3,1,2)
plt.plot(
	freqs[:2000],
	pspec[:2000]
)
plt.subplot(3,1,3)
plt.plot(recon)
plt.show()
