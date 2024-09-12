import torch as t

#sample inputs
x = t.tensor(2.,requires_grad=True)
y = t.tensor(3.,requires_grad=True)

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
