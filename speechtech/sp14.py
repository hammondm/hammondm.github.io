import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft,rfftfreq

plt.rcParams['savefig.dpi'] = 300

#one cycle of the pulse
period = 6

#crude approximation of the pulse
xs = [0,1,3,5,6]
ys = [0,0,1,0.9,0]

#100 cycles, 10*period samples per cycle
x = np.linspace(0,period*100,period*100*10)

#sample rate
sr = len(x)

#repeat pulse 100 times
pulses = np.interp(x,xs,ys,period=period)

#two plots
fig,axs = plt.subplots(2)
fig.suptitle('glottal pulse and spectrum')

#waveform
axs[0].plot(x[:1000],pulses[:1000])

#compute fft
yf = rfft(pulses)
xf = rfftfreq(sr,1/sr)

#plot fft
axs[1].plot(xf,np.abs(yf))
plt.show()
