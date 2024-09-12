import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq

#mel conversion and inverse
def f2m(x): return 2595 * np.log10(1 + x/700)
def m2f(m):
  return (np.power(10.0,m/2595) - 1) * 700

#some hz values
x = np.arange(0,10000)
#convert them all
m = f2m(x)

#try different values here
sr = 44100    #sample rate
samples = sr  #total number of samples
filters = 10  #number of filters
windows = 30  #number of time windows

#how many windows
windowsize = sr//windows
#wrt/ sample rate
xf = rfftfreq(windowsize,1/sr)
freqmax = xf.max()
binsize = freqmax//filters
mbinsize = f2m(freqmax)//filters

#loop through the windows
print('bins in hz, mel, mel->hz')
for i in range(filters+1):
	print(
		f'\t{binsize*i:>7.1f}' +
		f'\t{mbinsize*i:>7.1f}' +
		f'\t{m2f(mbinsize*i):>7.1f}'
)
