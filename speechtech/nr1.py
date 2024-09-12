import os

#file location
filedir = \
	'/data/cv-corpus-16.0-2023-12-06/cy/'
#get filenames
filenames = os.listdir(filedir+'clips')
#make new directory
os.system('mkdir ' + filedir + 'mhwav')

#go through all the files
for filename in filenames:
	newname = filename[:-3] + 'wav'
	#use ffmpeg to do conversions
	os.system(
		'ffmpeg -i ' + filedir + 'clips/' + \
		filename + ' ' + filedir + \
		'mhwav/' + newname
	)
