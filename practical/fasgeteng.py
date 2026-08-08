import re,os

dirname = 'mfapersian/'

filenames = [f for f in os.listdir(dirname) if \
	f.endswith('.txt')]

words = set()

for filename in filenames:
	f = open(dirname+filename,'r')
	t = f.read()
	f.close()
	if re.match('.*[a-zA-Z].*',t):
		print(f'{filename}: {t}')

