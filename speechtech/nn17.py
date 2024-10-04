import torch.nn as nn
import torch as t

emb = nn.Embedding(5,3)

#get the parameters
params = emb.named_parameters()
for name,param in params:
   print(f'{name}:\n{param}\n')

inp = t.tensor(2)

print(f'input {inp}\noutput: {emb(inp)}')

