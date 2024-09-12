import torch as t
import torch.nn as nn

#check for gpu
if t.cuda.is_available():
   dev = 'cuda'
else:
   dev = 'cpu'

epochs = 8000

#data for XOR
xs = t.Tensor([
	[0.,0.],
	[0.,1.],
	[1.,0.],
	[1.,1.]
]).to(dev)
y = t.Tensor([[0.],[1.],[1.],[0.]]).to(dev)

#define net
class XOR(nn.Module):
	def __init__(self):
		super(XOR,self).__init__()
		self.linear = nn.Linear(2,2)
		self.Sigmoid = nn.Sigmoid()
		self.linear2 = nn.Linear(2,1)
	def forward(self,inp):
		x = self.linear(inp)
		sig = self.Sigmoid(x)
		yh = self.linear2(sig)
		return yh

#move net to gpu
xor = XOR().to(dev)

#define loss function
mseloss = nn.MSELoss()
#stochastic gradient descent
optimizer = t.optim.SGD(
	xor.parameters(),
	lr=0.1
)

#train
for epoch in range(epochs):
	#output
	yhat = xor.forward(xs)
	#loss
	loss = mseloss(yhat,y)
	#backprop
	loss.backward()
	#update
	optimizer.step()
	#zero gradients
	optimizer.zero_grad()

#test
for i in range(xs.shape[0]):
	inp = xs[i,:]
	out = xor(inp)
	print(f'{inp}: {out.round().detach()}')
