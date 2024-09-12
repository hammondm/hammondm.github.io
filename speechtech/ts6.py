from scipy.io import wavfile
import matplotlib.pyplot as plt

fs,w = wavfile.read('mha.wav')

#plot highlighted cycle
plt.plot(range(1200),w[850:2050],'b')
plt.plot(range(200,1000),w[1050:1850],'b',linewidth=3)
plt.title('centered pulse unit')
plt.show()


