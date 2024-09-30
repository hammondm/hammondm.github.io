#tweaked from
#https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

import torch
import torch.nn as nn

#features,layers*features,layers
gru = nn.GRU(10,20,2)
#length,batch,features
inp = torch.randn(5,3,10)
#layers,batch,layers*features
h0 = torch.randn(2,3,20)

output,hn = gru(inp,h0)

