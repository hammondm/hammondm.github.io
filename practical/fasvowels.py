import parselmouth
import os
import numpy as np
from praatio import textgrid
from multiprocessing import Pool
import matplotlib.pyplot as plt

#get and plot average values for some vowel in whole corpus
#(parallel version)

vowel = 'a'

#file locations
tgpfx = 'persaligned/'
sndpfx = 'mfapersian/'

grids = [f for f in os.listdir(tgpfx) if f.endswith('TextGrid')]

def getvals(grid):
	vals = []
	wav = grid[:-9] + '.mp3'
	#read in sound and get formants
	snd = parselmouth.Sound(sndpfx+wav)
	formants = snd.to_formant_burg()
	#read in textgrid and get phones
	tg = textgrid.openTextgrid(tgpfx+grid,False)
	tier = tg.getTier('phones')
	#get formants and durations for all instances of [a]
	for i in tier:
		if i.label == vowel:
			#find midpoint
			midpoint = i.start + ((i.end - i.start)/2)
			f1 = formants.get_value_at_time(1,midpoint)
			f2 = formants.get_value_at_time(2,midpoint)
			f3 = formants.get_value_at_time(3,midpoint)
			dur = i.end-i.start
			vals.append((f1,f2,f3,dur))
	return vals

if __name__ == '__main__':
	f1s = []
	f2s = []
	f3s = []
	durs = []
	#get values in parallel
	with Pool() as p:
		res = p.map(getvals,grids)
	#flatten results
	res = [x for xs in res for x in xs]
	#separate results
	for (f1,f2,f3,dur) in res:
		if not np.isnan(f1): f1s.append(f1)
		if not np.isnan(f2): f2s.append(f2)
		if not np.isnan(f3): f3s.append(f3)
		durs.append(dur)
	#display
	print(f'vowel: {vowel}')
	print(f'{len(f1s)} samples')
	print(f'F1: {sum(f1s)/len(f1s)}')
	print(f'F2: {sum(f2s)/len(f2s)}')
	print(f'F3: {sum(f3s)/len(f3s)}')
	print(f'Duration: {sum(durs)/len(durs)}')

	plt.subplot(1,3,1)
	plt.subplots_adjust(wspace=.5)
	plt.gca().set_title(f'F1\n{np.mean(f1s):.2f}')
	plt.hist(f1s)
	plt.subplot(1,3,2)
	plt.gca().set_title(f'F2\n{np.mean(f2s):.2f}')
	plt.hist(f2s)
	plt.subplot(1,3,3)
	plt.gca().set_title(f'F3\n{np.mean(f3s):.2f}')
	plt.hist(f3s)
	plt.show()

