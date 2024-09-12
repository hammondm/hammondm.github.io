import torch
import torch.nn as nn

class PyramidLayer(nn.Module):
	def __init__(self,indim,outdim):
		super(PyramidLayer,self).__init__()
		self.blstm = nn.LSTM(
			indim,outdim,1,bidirectional=True
		)
	def forward(self,inp):
		output,(hn,cn) = self.blstm(inp)
		#pad if sequence length is odd
		if output.shape[0] % 2 != 0:
			zeros = torch.zeros(
				output.shape[1],
				output.shape[2]
			)
			zeros = zeros.unsqueeze(0)
			output = torch.vstack([output,zeros])
		#make pairs
		res = []
		i = 0
		while i < output.shape[0]:
			res.append(
				torch.hstack([output[i],output[i+1]])
			)
			i += 2
		output = torch.stack(res)
		return output,(hn,cn)

pblstm = PyramidLayer(6,7)

#random input
inp = torch.randn(5,3,6)

#calculate output
output,(hn,cn) = pblstm(inp)

#sequence,batchsize,2*hidden Fs
print('output shape:',output.shape)

