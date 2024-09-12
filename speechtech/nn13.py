import torch
import torch.nn as nn

#sample inputs
inp = torch.arange(1,21,dtype=torch.float)
inp = inp.view(2,1,-1)

#show
print('input:\n',inp)
print('\nshape:\n',inp.shape)

#apply batch normalization
bn = nn.BatchNorm1d(1)

#show
print('\nBatchNorm1d:\n',bn(inp))

#do it by hand
print(
	'\nBy hand:\n',
	(inp[0] - inp.mean()) \
	/inp.var(unbiased=False).sqrt()
)
