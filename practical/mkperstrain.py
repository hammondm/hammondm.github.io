import re,os,shutil

fileloc = 'cv-corpus-26.0-2026-06-12/fa/'

newdir = 'mfapersian/'

os.mkdir(newdir)

f = open(fileloc+'validated.tsv','r')
t = f.read()
f.close()

t = t.split('\n')
t = t[1:-1]

for line in t[:1000]:
	fields = line.split('\t')
	fname = fields[1]
	s = fields[3].lower()
	s = re.sub('[\–”‘\-\—\!¬“,\'"\.\?;:]',' ',s)
	s = s.strip()
	s = re.sub(' +',' ',s)
	newname = re.sub('\.mp3','.txt',fname)
	if not re.match('[a-zA-z]',s):
		g = open(newdir+newname,'w')
		g.write(s+'\n')
		g.close()
		shutil.copyfile(fileloc+'clips/'+fname,newdir+fname)

