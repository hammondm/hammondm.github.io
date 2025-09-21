from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['savefig.dpi'] = 300

fs,w = wavfile.read('mha.wav')

#one segment
w1 = w[1050:1850]
#apply hanning window
h = np.hanning(len(w1))
w1 = w1 * h
#concatenate
w3 = np.zeros(1301)
w3[1:601] = w1[200:800]
w3[501:1301] = w3[501:1301] + w1
#plot
plt.plot(w3)
plt.title('concatenated/added (lower pitch)')
plt.show()
