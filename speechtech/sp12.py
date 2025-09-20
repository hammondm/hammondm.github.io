import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft,rfftfreq

plt.rcParams['savefig.dpi'] = 300

vals = np.arange(0,200*np.pi,.4)

hlen = len(vals)

print(f'length and sample rate: {hlen}')

#all components under Nyquist rate
hs1 = np.zeros(hlen)
for i in range(5,8):
	h = np.sin(i*vals)
	hs1 += h

#some components over Nyquist rate
hs2 = np.zeros(hlen)
for i in range(5,10):
	h = np.sin(i*vals)
	hs2 += h

#get the frequency slots
w = rfftfreq(hlen,d=1/hlen)

#compute ffts
rspec1 = rfft(hs1)
rspec2 = rfft(hs2)

#plot accurate components
plt.subplot(2,1,1)
plt.plot(w,np.abs(rspec1))
#plot aliased components
plt.subplot(2,1,2)
plt.plot(w,np.abs(rspec2))
plt.show()

