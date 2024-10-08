import torch.nn as nn
import torch

class SEblock(nn.Module):
	def __init__(self,c,r=16):
		super().__init__()
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excite = nn.Sequential(
			nn.Linear(c,c // r,bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(c // r,c,bias=False),
			nn.Sigmoid()
		)
	def forward(self,x):
		bs,c,_,_ = x.shape
		y = self.squeeze(x).view(bs,c)
		y = self.excite(y).view(bs,c,1,1)
		return x * y.expand_as(x)

seb = SEblock(100)

t = torch.rand([10,100,3,40])

res = seb.forward(t)

print(res.shape)

