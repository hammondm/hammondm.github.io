import numpy as np
import torch as t
import torch.nn.functional as F

#input
inp = t.tensor([5.,2.,7.])

print('input:',inp)
#softmax by hand
print('softmax:',(inp.exp()/inp.exp().sum()))
print(
	'log softmax by hand:',
	(inp.exp()/inp.exp().sum()).log()
)
#softmax with torch
print('log softmax:',F.log_softmax(inp,dim=0))
