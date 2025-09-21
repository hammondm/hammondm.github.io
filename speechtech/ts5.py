from scipy.io import wavfile
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

fs,w = wavfile.read('mha.wav')

#plot two cycles
plt.plot(w[1250:2050])
plt.title('two cycles')
plt.show()

