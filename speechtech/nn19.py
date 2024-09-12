import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt

class Att(nn.Module):
	def __init__(self,edim,dedim):
		super().__init__()
		#between encoder and decoder
		self.edim = edim
		self.dedim = dedim
		#scaling factor
		self.v = nn.Parameter(
			#random numbers between -.1 and .1
			t.FloatTensor(self.dedim).uniform_(
				-0.1,0.1
			)
		)
		self.W1 = nn.Linear(self.dedim,self.dedim)
		self.W2 = nn.Linear(self.edim,self.dedim)
	def forward(self,query,values): 
		#get the weights
		weights = self.getweights(query,values)
		#softmax over weights
		weights = nn.functional.softmax(
			weights,
			dim=0
		)
		#dot product of weights and values
		return weights @ values
	def getweights(self,query,values):
		query = query.repeat(values.size(0),1)
		weights = self.W1(query) + self.W2(values)
		return t.tanh(weights) @ self.v

#encoder dimension
edim = 100
#decoder dimension
dedim = 50
#length of input
encseqlen = 10
#length of output
decseqlen = 15
#make the attention NN
att = Att(edim,dedim)
#random input
enchidden = t.rand(encseqlen,edim)
#random output
dechidden = t.rand(decseqlen,dedim)
#attention weights 
weights = t.FloatTensor(decseqlen,encseqlen)

#chunk through an output
for step in range(decseqlen):
	#this invokes forward()
	_ = att(dechidden[step],enchidden)
	weights[step] = att.getweights(
		dechidden[step],
		enchidden
	)

#display attention as heatmap
plt.imshow(
	weights.detach().numpy(),
	cmap='hot',
	interpolation='nearest'
)
plt.show()

