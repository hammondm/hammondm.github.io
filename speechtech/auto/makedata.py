import re

#variables to specify how much data
testlen = 3000
validlen = 3000

#read corpus
f = open('browncorpus.txt','r')
t = f.read()
f.close()

t = t.split('\n')

#normalizing function
def dolines(n,m,filename):
	f = open(filename,'w')
	for line in t[n:m]:
		line = re.sub(
			'([^ ])([\.,])',r'\1 \2',
			line
		)
		line = re.sub("'s"," 's",line)
		line = re.sub("'t"," 't",line)
		line = re.sub(';',' ; ',line)
		line = re.sub(':',' : ',line)
		line = re.sub('"',' " ',line)
		line = re.sub('  *',' ',line)
		f.write(line+'\n')
	f.close()

#split into test,validation,train
dolines(0,testlen,'test.txt')
dolines(testlen,testlen+validlen,'valid.txt')
dolines(testlen+validlen,len(t),'train.txt')
