import torch.nn as nn
import torch.nn.functional as F
import torch

class Highway(nn.Module):
	def __init__(self,size,numlayers):
		super(Highway,self).__init__()
		#how many layers
		self.numlayers = numlayers
		#normal weights
		self.nonlinear = nn.ModuleList(
			[nn.Linear(size,size) \
				for _ in range(numlayers)]
		)
		#carry gate weights
		self.linear = nn.ModuleList(
			[nn.Linear(size,size) \
				for _ in range(numlayers)]
		)
		#transform gate weights
		self.gate = nn.ModuleList(
			[nn.Linear(size,size) \
				for _ in range(numlayers)]
		)
	def forward(self,x):
		#go through layer by layer
		for layer in range(self.numlayers):
			gate = torch.sigmoid(
				self.gate[layer](x)
			)
			nonlinear = F.relu(
				self.nonlinear[layer](x)
			)
			linear = self.linear[layer](x)
			x = gate * nonlinear + \
				(1 - gate) * linear
		return x

if __name__ == '__main__':
	#itemlength=5, layers=3
	highway = Highway(5,3)
	#batch of 4, itemlength=5
	i = torch.rand(4,5)
	print('input:\n',i,sep='')
	#generate output
	o = highway(i)
	print('\noutput:\n',o,sep='')

