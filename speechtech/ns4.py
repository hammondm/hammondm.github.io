vallen = 10
testlen = 10
trainlen = 100

#read LJSpeech metadata file
f = open('metadata.csv','r')
t = f.read()
f.close()

#break into lines
t = t.split('\n')
print(len(t))

#write training metadata file
f = open('train.csv','w')
for line in t[:trainlen]:
	f.write(line+'\n')
f.close()

#write validation metadata file
f = open('val.csv','w')
for line in t[trainlen:trainlen+vallen]:
	f.write(line+'\n')
f.close()

#calculate indices for test data
startpoint = trainlen+vallen
endpoint = trainlen+vallen+testlen

#write test metadata file
f = open('test.csv','w')
for line in t[startpoint:endpoint]:
	f.write(line+'\n')
f.close()
