from scipy.signal import sawtooth
import numpy as np
from scipy.io import wavfile

#sample rate, frequency, length
sr = 10000
freq = 100
length = 500

#source wave
source = sawtooth(
	2*np.pi*freq*np.arange(length)/sr
)

#variables from equations
ff = 1000
fbw = 50
t = 1/sr

#just as in text
c = -np.exp(-2 * np.pi * fbw * t)
b = 2 * np.exp(-np.pi * fbw * t) * \
	np.cos(2*np.pi*ff * t)
a = (1 - b) - c

#step through the signal
res = []
for v in range(len(source)):
	prev = v-1
	prevprev = v-2
	val = a * source[v]
	if prev >= 0:
		val += b * res[-1]
	if prevprev >= 0:
		val += c * res[-2]
	res.append(val)

res = np.array(res)

wavfile.write('source.wav',sr,source)
wavfile.write('res.wav',sr,res)
