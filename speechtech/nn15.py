#tweaked from
#https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

import torch
import torch.nn as nn

#input features,hidden size,layers
lstm = nn.LSTM(4,2,1)
#input length,batchsize,input features
inp = torch.randn(5,3,4)
#D*layers,batchsize,output features
h0 = torch.randn(1,3,2)
#same
c0 = torch.randn(1,3,2)

output,(hn,cn) = lstm(inp,(h0,c0))

