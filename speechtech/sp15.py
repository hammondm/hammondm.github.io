import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft,rfftfreq

plt.rcParams['savefig.dpi'] = 300

hlen = len(np.arange(0,200*np.pi,.1))
hs = np.zeros(hlen)

amp = 1
for i in range(1,31):
	h = np.sin(i*np.arange(0,200*np.pi,.1))
	h *= amp
	amp -= .01
	hs += h

#get the frequency slots
w = rfftfreq(hlen,d=1/hlen)

#compute the fft
rspec = rfft(hs)

plt.subplot(2,1,1)
plt.plot(hs[:800])
plt.subplot(2,1,2)
plt.plot(w,np.abs(rspec))
plt.show()
