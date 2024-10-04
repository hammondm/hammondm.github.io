import torch as t
import torch.nn as nn

#log softmax nodes
m = nn.LogSoftmax(dim=1)
#negative log likelihood loss
loss = nn.NLLLoss()

#random input
inp = t.randn(4,5)
print('input:\n',inp)
#target
target = t.tensor([2,1,3,2])
print('target:\n',target)
#current output
output = m(inp)
print('output:\n',output)
#calculate loss
print('loss:\n',loss(output,target))

#loss by hand
val = 0
for n,t in enumerate(target):
	val += -1 * output[n,t]

print(val/len(target))
