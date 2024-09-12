import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#read in wave
fs,w = wavfile.read('mha16000.wav')

#25msec windows
framesize = int(fs * .025)
frame = w[2000:2000+framesize]

#make a hamming window
hw = np.hamming(framesize)

plt.subplot(3,1,1)
plt.plot(frame)
plt.subplot(3,1,2)
plt.plot(hw)
plt.subplot(3,1,3)
plt.plot(hw*frame)
plt.show()
