from fastrans import dotrans
import re

f = open('wikipronfas.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

for line in lines:
	bits = line.split('\t')
	print(f'{bits[0]}\t({re.sub(' ','',bits[1])})\t{dotrans(bits[0])}')

