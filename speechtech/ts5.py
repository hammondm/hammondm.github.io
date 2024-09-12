from scipy.io import wavfile
import matplotlib.pyplot as plt

fs,w = wavfile.read('mha.wav')

#plot two cycles
plt.plot(w[1250:2050])
plt.title('two cycles')
plt.show()

