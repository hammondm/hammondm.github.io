import re
import matplotlib.pyplot as plt

#plotting aed values

f = open('wg2p.py','r')
t = f.read()
f.close()

lines = t.split('\n')

vals = []
for line in lines:
	if re.search('^#total =',line):
		num = re.sub('^#total = ','',line)
		num = re.sub(' .*','',num)
		vals.append(int(num))

diffs = []
for i in range(1,len(vals)):
	diffs.append(vals[i-1]-vals[i])

plt.subplot(1,2,1)
plt.plot(vals)
plt.title('aed values')
plt.subplot(1,2,2)
plt.plot(diffs)
plt.title('raw improvement')
plt.show()

#plt.subplots_adjust(wspace=.5)

