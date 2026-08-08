import re

def fixlines(filename):
	f = open(filename,'r')
	t = f.read()
	f.close()

	#remove foreign characters and empty final line
	lines = t.split('\n')[:-10]

	newlines = []
	for line in lines:
		fixedline = re.sub("[ˌˈ]","",line)
		fixedline = re.sub(" \\.","",fixedline)
		fixedline = re.sub("kᵊ","k ə",fixedline)
		fixedline = re.sub('ɨ̞','ɨ',fixedline)
		fixedline = re.sub('i̞','i',fixedline)
		fixedline = re.sub('r̊','r̥',fixedline)
		newlines.append(fixedline)
	return newlines

if __name__ == "__main__":
	lines = fixlines('wikicym2.txt')
	for line in lines:
		print(line)

