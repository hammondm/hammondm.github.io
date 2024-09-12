from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

fs,w = wavfile.read('mha.wav')

#one segment
w1 = w[1050:1850]
#apply hanning window
h = np.hanning(len(w1))
w1 = w1 * h
#concatenate
w3 = np.zeros(1201)
w3[1:601] = w1[200:800]
w3[401:1201] = w3[401:1201] + w1
#plot
plt.plot(w3)
plt.title('concatenated/added (same pitch)')
plt.show()
