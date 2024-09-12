import numpy as np
import torch
import torch.nn as nn

in1d = torch.arange(1,11,dtype=torch.float)

print('1d input:\n',in1d)
print('shape:\n',in1d.shape,'\n')
in1d = in1d.unsqueeze(0).unsqueeze(0)
print('input transformed:\n',in1d)
print('transormed shape:\n',in1d.shape)

#set up the convolution
cnn1d_1 = nn.Conv1d(
	in_channels=1,
	out_channels=1,
	kernel_size=3,
	stride=1
)

#set weights and bias
weight = torch.tensor(
	[[[1.,0.,1.]]],
	requires_grad=True
)
bias = torch.tensor([0.],requires_grad=True)
with torch.no_grad():
	cnn1d_1.weight = nn.Parameter(weight)
	cnn1d_1.bias = nn.Parameter(bias)

#print parameters
print('\nparameters:')
params = cnn1d_1.named_parameters()
for name,param in params:
	print(
		'\t',
		name,
		': ',
		param.detach().numpy(),
		sep=''
	)

print("\noutput:\n",cnn1d_1(in1d))

#maxpool
mp = nn.MaxPool1d(3,2)

print('\nmaxpool input:\n',in1d)
print('maxpool output:\n',mp(in1d))

