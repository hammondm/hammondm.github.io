import librosa,librosa.display
import numpy as np
import matplotlib.pyplot as plt

y,sr = librosa.load('quick.wav')
#magnitude spectrogram
S = np.abs(librosa.stft(y))
#invert
y_inv = librosa.griffinlim(S)

#plot
plt.subplot(3,1,1)
plt.plot(y)
plt.title('original')
plt.subplot(3,1,2)
db = librosa.amplitude_to_db(S,ref=np.max)
librosa.display.specshow(
	db,
	sr=sr,
	cmap='gray_r'
)
plt.subplot(3,1,3)
plt.plot(y_inv)
plt.title('reconstructed')
plt.show()
