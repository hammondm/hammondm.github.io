import torch
import torch.nn as nn

#GLU model
class GLU(nn.Module):
	def __init__(self,size):
		super().__init__()
		self.lin1 = nn.Linear(size,size)
		self.lin2 = nn.Linear(size,size)
	def forward(self,inp):
		#math as in text
		res = self.lin1(inp) * \
			self.lin2(inp).sigmoid()
		return res

#initialize
glu = GLU(10)

#print result
print(glu(torch.Tensor(3,10,10)).shape)
