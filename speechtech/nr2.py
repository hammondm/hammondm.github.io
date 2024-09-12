import warnings,torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import \
	Dataset,DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

#ignore 'divide by zero' warnings
warnings.filterwarnings("ignore")

processes = 15
datadir = '/data/cv-corpus-16.' + \
	'0-2023-12-06/cy/'
wavdir = datadir + 'mhwav/'
validated = 'validated.tsv'
files = 300
valid = 20
test = 20
batchsize = 5
inputdim = 128
hiddendim = 500
layers = 5
lr = 0.01
epochs = 5
bidir = True

#use GPU if available
if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
print(f'using {device}')

#map from letters to integers
def addtol2i(s,d):
	for letter in s:
		if letter not in d:
			#reserve 0 for blank
			d[letter] = len(d)+1

#get spectrogram and target
def getspec(pair):
	trans = [l2i[let] for let in pair[1]]
	filename = wavdir+pair[0][:-3]+'wav'
	fs,w = wavfile.read(filename)
	res = plt.specgram(
		w,
		NFFT=254,
		Fs=fs,
		noverlap=127
	)
	spec = res[0].T
	return (trans,spec)

#read in list of files
f = open(datadir+validated,'r')
t = f.read()
f.close()

#remove header and empty final line
t = t.split('\n')
t = t[1:-1]

#get filenames,glosses, and character map
l2i = {}
pairs = []
for line in t:
	bits = line.split('\t')
	filename = bits[1]
	gloss = bits[2]
	if gloss[0] == '"': gloss = gloss[1:]
	if gloss[-1] == '"': gloss = gloss[:-1]
	addtol2i(gloss,l2i)
	pairs.append((filename,gloss))

#make reverse map from integers to characters
i2l = \
	{pair[1]:pair[0] for pair in l2i.items()}

#number of output categories (wo/blank!)
outsize = len(i2l)

#make spectrograms in parallel
with mp.Pool(processes) as mypool:
	results = mypool.map(getspec,pairs[:files])

#separate training, validation, test
validset = results[:valid]
testset = results[valid:valid+test]
trainset = results[valid+test:]

#custom dataset
class SpecData(Dataset):
	def __init__(self,d):
		self.labels = [torch.Tensor(pair[0]) \
			for pair in d]
		self.specs = [torch.Tensor(pair[1]) \
			for pair in d]
	def __len__(self):
		return len(self.labels)
	def __getitem__(self,idx):
		spec = self.specs[idx]
		label = self.labels[idx]
		return spec,label

#make datasets
traindata = SpecData(trainset)
testdata = SpecData(testset)
validdata = SpecData(validset)

#items in batch must have same length
def pad(batch):
	(xx,yy) = zip(*batch)
	xlens = [len(x) for x in xx]
	ylens = [len(y) for y in yy]
	xxpad = pad_sequence(
		xx,
		batch_first=True,
		padding_value=0
	)
	yypad = pad_sequence(
		yy,
		batch_first=True,
		padding_value=0
	)
	return xxpad,yypad,xlens,ylens

#make dataloaders
trainloader = DataLoader(
	traindata,
	batch_size=batchsize,
	collate_fn=pad,
	shuffle=True
)
validloader = DataLoader(
	validdata,
	batch_size=batchsize,
	collate_fn=pad,
	shuffle=True
)
#batch = 1 for test
testloader = DataLoader(
	testdata,
	batch_size=1,
	shuffle=False
)

#NN with LSTMs and logsoftmax
class ASR(nn.Module):
	def __init__(
			self,idim,hdim,numlayers,osize
		):
		super(ASR,self).__init__()
		self.hdim = hdim
		self.lstm = nn.LSTM(
			idim,
			hdim,
			numlayers,
			bidirectional=bidir,
			batch_first=True
		)
		if bidir:
			self.hidden2out = \
				nn.Linear(hdim*2,osize)
		else:
			self.hidden2out = nn.Linear(hdim,osize)
	def forward(self,inp):
		lstm_out,_ = self.lstm(inp)
		outlin = self.hidden2out(lstm_out)
		scores = F.log_softmax(outlin,dim=2)
		return scores

asr = ASR(
	inputdim,
	hiddendim,
	layers,
	outsize+1
).to(device)
lossfunc = nn.CTCLoss(
	#zero_infinity=True,
	reduction='mean'
)
opt = optim.SGD(
	asr.parameters(),
	lr=lr
)

#train
for epoch in range(epochs):
	i = 0
	epochloss = []
	for inp,outp,inlens,outlens in trainloader:
		asr.zero_grad()
		inp = inp.to(device)
		pred = asr(inp)
		loss = lossfunc(
			pred.transpose(1,0),
			outp,
			inlens,
			outlens
		)
		loss.backward()
		opt.step()
		epochloss.append(
			loss.detach().cpu().numpy()
		)
		i += 1
	elossmean = np.mean(epochloss)
	print(f'epoch {epoch} loss: {elossmean}')
	#validate
	with torch.no_grad():
		validloss = []
		for inp,outp,inlens,outlens in \
				validloader:
			inp = inp.to(device)
			pred = asr(inp)
			loss = lossfunc(
				pred.transpose(1,0),
				outp,
				inlens,
				outlens
			)
			validloss.append(
				loss.detach().cpu().numpy()
			)
		print(
			f'\tvalid loss: {np.mean(validloss)}'
		)

#test one at a time
with torch.no_grad():
	for inp,outp in testloader:
		for i in outp[0]:
			print(i2l[int(i)],end='')
		print()
		inp = inp.to(device)
		outp = outp.to(device)
		pred = asr(inp)
		res = \
			pred.squeeze().detach().cpu().numpy()
		#greedy decoding does NOT work well!
		res = res.argmax(axis=1)
		#eliminate duplicates
		newres = [res[0]]
		for n in res[1:]:
			if n != newres[-1]:
				newres.append(n)
		#eliminate blanks
		newres = [i2l[n] for n in newres \
			if n != 0]
		print(f'"{"".join(newres)}"',end='\n\n')

