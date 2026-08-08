from praatio import textgrid
import parselmouth
import os
import numpy as np
from parselmouth.praat import call

#get formants and durations for whole corpus

#file and locations
tgpfx = 'mfaaligned2/'
sndpfx = 'mfawelsh2/'

grids = [f for f in os.listdir(tgpfx) if \
	f.endswith('TextGrid')]

print('filename\tphoneme\tF0\tF1\tF2\tF3\tduration')

for grid in grids:
	wav = grid[:-9] + '.wav'
	#read in sound and get formants
	snd = parselmouth.Sound(sndpfx+wav)
	formants = snd.to_formant_burg()
	pitch = snd.to_pitch()
	#read in textgrid and get phones
	tg = textgrid.openTextgrid(tgpfx+grid,False)
	tier = tg.getTier('phones')
	#get formants and durations for all instances of [a]
	for i in tier:
		#find midpoint
		midpoint = i.start + ((i.end - i.start)/2)
		f1 = formants.get_value_at_time(1,midpoint)
		f2 = formants.get_value_at_time(2,midpoint)
		f3 = formants.get_value_at_time(3,midpoint)
		dur = i.end-i.start
		p = call(
			pitch,
			"Get value at time",
			midpoint,
			"Hertz",
			"Linear"
		)
		print(f'{wav}\t{i.label}\t',end='')
		print(f'{p:.3f}\t{f1:.3f}\t{f2:.3f}\t',end='')
		print('{f3:.3f}\t{dur:.3f}')

