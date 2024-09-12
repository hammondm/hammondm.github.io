import scipy.io.wavfile as wavfile
from world import main
import numpy as np

#read in wavefile
fs,x_int16 = wavfile.read('mha.wav')
x = x_int16/(2**15-1)
#extract spectrum and pitch
vocoder = main.World()
dat = vocoder.encode(fs,x,f0_method='harvest')
#raise the pitch
dat = vocoder.scale_pitch(dat,1.5)
#put it back together
dat = vocoder.decode(dat)
#save to a file
wavfile.write(
	'mhaHIGHER.wav',
	fs,
	(dat['out'] * 2 ** 15).astype(np.int16)
)

