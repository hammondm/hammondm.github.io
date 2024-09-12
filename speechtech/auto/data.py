import os,torch
from io import open

#maps words/indexes
class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
	def add_word(self,word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(
				self.idx2word
			) - 1
		return self.word2idx[word]
	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self,path):
		#general mapping
		self.dictionary = Dictionary()
		#training data
		self.train = self.tokenize(
			os.path.join(path,'train.txt')
		)
		#validation data
		self.valid = self.tokenize(
			os.path.join(path,'valid.txt')
		)
		#test data
		self.test = self.tokenize(
			os.path.join(path,'test.txt')
		)
	def tokenize(self,path):
		assert os.path.exists(path)
		#first, add words to dictionary
		with open(path,'r',encoding="utf8") as f:
			#each line is a sentence
			for line in f:
				words = line.split() + ['<eos>']
				for word in words:
					self.dictionary.add_word(word)
		#then actually tokenize file content
		with open(path,'r',encoding="utf8") as f:
			idss = []
			for line in f:
				words = line.split() + ['<eos>']
				ids = []
				for word in words:
					ids.append(
						self.dictionary.word2idx[word]
					)
				idss.append(
					torch.tensor(ids).type(torch.int64)
				)
			ids = torch.cat(idss)
		#everything concatenated in one big tensor
		return ids

