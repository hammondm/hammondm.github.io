import os

#file location
filedir = 'cv-corpus-19.0-2024-09-13/cy/'

newsubdir = 'mhwav'

#get filenames
filenames = os.listdir(filedir+'clips')
#make new directory
os.system('mkdir ' + filedir + newsubdir)

#go through all the files
for filename in filenames:
	newname = filename[:-3] + 'wav'
	#use ffmpeg to do conversions
	os.system(
		'ffmpeg -i ' + filedir + 'clips/' + \
		filename + ' ' + filedir + \
		'mhwav/' + newname
	)

