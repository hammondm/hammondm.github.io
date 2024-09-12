from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

fs,w = wavfile.read('mha.wav')

#plot hanning window
w1 = w[1050:1850]
h = np.hanning(len(w1))
plt.plot(w1,'b--')
w1 = w1 * h
plt.plot(w1,'b')
plt.title('hanning window')
plt.show()


