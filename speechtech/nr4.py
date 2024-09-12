import torch
import torch.nn as nn

#make a bidirectional LSTM
blstm = nn.LSTM(6,7,1,bidirectional=True)

#random input
inp = torch.randn(5,3,6)

#calculate output
output,(hn,cn) = blstm(inp)

#sequence,batchsize,2*hidden Fs
print('output shape:',output.shape)

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

#new shape
print('new output shape:',output.shape)

