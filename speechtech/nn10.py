import torch as t

#gpu-aware version
if t.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'

#sample inputs
x = t.tensor(
	2.,
	requires_grad=True,
	device=dev
)
y = t.tensor(
	3.,
	requires_grad=True,
	device=dev
)

#do some math
z = 3*x + y

#show results
print('x =',x)
print('y =',y)
print('3*x + y =',z)

#calculate differentials
z.backward()

#show results
print('dx =',x.grad)
print('dy =',y.grad)
