import re,os,shutil

fileloc = 'cv-corpus-19.0-2024-09-13/cy/'

#location of new corpus
newdir = 'mfawelsh/'
os.mkdir(newdir)

#read metadata
f = open(fileloc+'validated.tsv','r')
t = f.read()
f.close()

#break into lines
t = t.split('\n')
t = t[1:-1]

#use only 1000 files
for line in t[:1000]:
	#get the wav file name
	fields = line.split('\t')
	fname = fields[1]
	#normalize the transcript
	s = fields[3].lower()
	s = re.sub('[\–”‘\-\—\!¬“,\'"\.\?;:]',' ',s)
	s = s.strip()
	s = re.sub(' +',' ',s)
	#put each transcript in its own file
	newname = re.sub('\.mp3','.txt',fname)
	#use wav files, not mp3 files
	wavname = re.sub('\.mp3','.wav',fname)
	#write transcript
	g = open(newdir+newname,'w')
	g.write(s+'\n')
	g.close()
	#copy wav files
	shutil.copyfile(fileloc+'mhwav/'+wavname,newdir+wavname)

