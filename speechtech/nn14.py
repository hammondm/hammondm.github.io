import torch
import torch.nn as nn

#create a 1 -> 1 RNN
rnn = nn.RNNCell(1,1)

#get the parameters
params = rnn.named_parameters()
for name,param in params:
	print(f'\t{name}: {param.data.item():.3}')

#two strings of values to go through
valueset = [
	[2.,1.,3.,2.],
	[2.,3.,3.,2.],
]

for values in valueset:
	print('input:',values)
	#initial hidden value
	h = torch.tensor([[0.]])
	#go through the values 1 by 1
	for value in values:
		#make a tensor
		x = torch.tensor([[value]])
		#input and last output as input
		h = rnn(x,h)
		print(f'\t{x.item():.3}: {h.item():.3}')

