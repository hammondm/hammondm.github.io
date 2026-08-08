import re,os

dirname = 'mfapersian/'

filenames = [f for f in os.listdir(dirname) if \
	f.endswith('.txt')]

words = set()

for filename in filenames:
	f = open(dirname+filename,'r')
	t = f.read()
	f.close()
	s = re.sub('[?,]',' ',t[:-1])
	s = s.strip()
	s = re.split(' +',s)
	for word in s:
		words.add(word)

for word in words:
	print(word)

