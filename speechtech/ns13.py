import numpy as np
import torch
import torch.nn as nn

in1d = torch.arange(1,11,dtype=torch.float)

print('1d input:\n',in1d)
print('shape:\n',in1d.shape,'\n')
in1d = in1d.unsqueeze(0).unsqueeze(0)
print('input transformed:\n',in1d)
print('transormed shape:\n',in1d.shape)

#set up transpose convolution
cnt1d_1 = nn.ConvTranspose1d(
	in_channels=1,
	out_channels=1,
	kernel_size=3,
	dilation=1,
	stride=1
)

#set weights and bias
weight = torch.tensor(
	[[[1.,0.,1.]]],
	requires_grad=True
)
bias = torch.tensor([0.],requires_grad=True)
with torch.no_grad():
	cnt1d_1.weight = nn.Parameter(weight)
	cnt1d_1.bias = nn.Parameter(bias)

print('\nparameters:')
params = cnt1d_1.named_parameters()
for name,param in params:
	pval = param.detach().numpy()
	print(f'\t{name}: {pval}')

print("\noutput:\n",cnt1d_1(in1d))

