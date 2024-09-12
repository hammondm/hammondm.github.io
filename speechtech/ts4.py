from scipy.io import wavfile
import matplotlib.pyplot as plt

fs,w = wavfile.read('mha.wav')
#plot one cycle
plt.plot(w[1250:1650])
plt.title('one cycle')
plt.show()

