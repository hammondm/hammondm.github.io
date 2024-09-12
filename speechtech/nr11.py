import torch
import torch.nn as nn

#make swish layer
m = nn.SiLU()
#random input
inp = torch.randn(2)
#generate output
outp = m(inp)

#print
print(inp)
print(outp)
