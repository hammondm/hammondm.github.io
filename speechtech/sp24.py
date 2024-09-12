import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.fft import rfft

fs,w = wavfile.read('quick.wav')
filters = 10
span = 1102

#spectrum for each window
i = 0
spectra = []
while i < len(w):
	if i + span < len(w):
		win = w[i:i+span]
		spectrum = rfft(win)
		spectra.append(spectrum)
	i += span

#compute value for each freq range
#max or mean, here max
binsize = len(spectra[0])//filters
fbanks = []
for spectrum in spectra:
	fbank = np.zeros(filters)
	i = 0
	while i < filters:
		fbank[i] = np.abs(
			spectrum[i:i+binsize]
		).max()
		i += 1
	fbanks.append(fbank)
fbanks = np.array(fbanks)

plt.imshow(
	fbanks.T,
	cmap='gray_r',
	aspect='auto'
)
plt.show()
