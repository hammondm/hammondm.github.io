#bigram demo

import re

#read in file
f = open('alice.txt','r')
t = f.read()
f.close()

#remove gutenberg header
t = t[1468:]

print(t[:100])

#convert to lowercase
t = t.lower()

#split into sentences
t = re.split('[\\.\?!]',t)

#remove other punctuation
t = [re.sub('[^a-z]',' ',line) for line in t]
#pad lines
t = ['# ' + line + ' #' for line in t]
#remove extra spaces
t = [re.sub(' +',' ',line) for line in t]

wordcounts = {}
bigramcounts = {}

#go through all the lines
for line in t:
	#break into words
	words = line.split(' ')
	#count word types
	for word in words:
		if word in wordcounts:
			wordcounts[word] += 1
		else:
			wordcounts[word] = 1
	#count bigram types
	for i in range(len(words) - 1):
		bigram = words[i] + ' ' + words[i+1]
		if bigram in bigramcounts:
			bigramcounts[bigram] += 1
		else:
			bigramcounts[bigram] = 1

#convert word counts to probabilities
wordtotal = sum(wordcounts.values())
wordprobs = {word:wordcounts[word]/wordtotal \
	for word in wordcounts}

#convert bigram counts to probabilities
bigramtotal = sum(bigramcounts.values())
bigramprobs = {bg:bigramcounts[bg]/ \
	bigramtotal for bg in bigramcounts}

#test sentences
ss = [
	'# i see you #',
	'# you see i #'
]

#do comparison
for s in ss:
	print(s)
	total = 1
	#break into words
	words = s.split(' ')
	for i in range(len(words) - 1):
		bigram = words[i] + ' ' + words[i+1]
		val = 0
		if bigram in bigramprobs:
			val = bigramprobs[bigram]
			val /= wordprobs[words[i]]
		total *= val
		print(f'\t{bigram}: {val}')
	print(f'\t{total}')
