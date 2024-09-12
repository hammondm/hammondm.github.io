import math,torch
import torch.nn as nn
import numpy as np

maxint=128
batchsize=16
steps=1000

#convert numbers to binary (without pfx)
def makeBinary(number):
	return [int(x) for x in \
		list(bin(number))[2:]]

#convert matrix of floats to list of ints
def makeInt(fmatrix,thresh=0.5):
	return [
		int("".join([str(int(y)) for y in x]),2) \
		for x in fmatrix >= thresh
	]

#make a bunch of even numbers
def makeEvens(maxint,batchsize=16):
	#get size to represent maximum number
	maxlength = int(math.log(maxint,2))
	#sample of integers in range 0-maxint
	samples = np.random.randint(
		0,
		int(maxint / 2),
		batchsize
	)
	#list of labels, all ones
	labels = [1] * batchsize
	#generate list of binary numbers
	data = [makeBinary(int(x * 2)) \
		for x in samples]
	data = [([0]*(maxlength-len(x)))+x \
		for x in data]
	return labels,data

#generator, one layer with sigmoid
class Generator(nn.Module):
	def __init__(self,ilnth):
		super(Generator,self).__init__()
		self.dense = nn.Linear(ilnth,ilnth)
		self.activation = nn.Sigmoid()
	def forward(self,x):
		return self.activation(self.dense(x))

#discriminator, one layer with sigmoid
class Discriminator(nn.Module):
	def __init__(self,ilnth):
		super(Discriminator,self).__init__()
		self.dense = nn.Linear(ilnth,1);
		self.activation = nn.Sigmoid()
	def forward(self,x):
		return self.activation(self.dense(x))

#TRAIN
inputlength = int(math.log(maxint,2))
#make models
generator = Generator(inputlength)
discriminator = Discriminator(inputlength)
#optimizers for both
genOpt = torch.optim.Adam(
	generator.parameters(),
	lr=0.001
)
discOpt = torch.optim.Adam(
	discriminator.parameters(),
	lr=0.001
)
#loss function
loss = nn.BCELoss()

#iterate
for i in range(steps):
	#zero gradients
	genOpt.zero_grad()
	#noisy input for generator
	noise = torch.randint(
		0,2,size=(batchsize,inputlength)
	).float()
	genData = generator(noise)
	#generate examples of even real data
	trueLabels,trueData = makeEvens(
		maxint,
		batchsize=batchsize
	)
	#convert to tensors
	trueLabels = torch.tensor(
		trueLabels
	).float()
	trueData = torch.tensor(trueData).float()
	trueLabels = trueLabels.unsqueeze(dim=1)
	#train generator
	genDiscOut = discriminator(genData)
	genLoss = loss(genDiscOut,trueLabels)
	genLoss.backward()
	genOpt.step()
	#zero discrimiator gradients
	discOpt.zero_grad()
	#train discriminator on true/generated data
	trueDiscOut = discriminator(trueData)
	trueDiscLoss = loss(trueDiscOut,trueLabels)
	genDiscOut = discriminator(genData.detach())
	genDiscLoss = loss(
		genDiscOut,
		torch.zeros(batchsize).unsqueeze(dim=1)
	)
	discLoss = (trueDiscLoss + genDiscLoss) / 2
	discLoss.backward()
	discOpt.step()
	if (i+1) % 100 == 0:
		with torch.no_grad():
			noise = torch.randint(
				0,2,size=(batchsize,inputlength)
			).float()
			genData = generator(noise)
			print(f'{i+1}:\t',makeInt(genData))

