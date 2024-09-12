import torch as t

#sequence of numbers
X = t.arange(8,dtype=t.float)
X = X.reshape([1,1,8])
print(f'initial sequence:\n\t{X}')

#make the convolution
H = t.ones([1,1,3])
Y = t.nn.Conv1d(3,1,1,stride=1,bias=False)
Y.weight = t.nn.Parameter(H)

#normal convolution
print(
	f'normal convolution:\n\t{Y(X).detach()}'
)

#pad with initial zeros
zeros = t.tensor([0.,0.])
zeros = zeros.reshape([1,1,2])
Xpad = t.cat([zeros,X],dim=2)
print(f'padded:\n\t{Xpad}')
print(
	'causal convolution:\n' +
	f'\t{Y(Xpad).detach()}'
)

#split into two
Z1 = X.reshape(4,-1).T
#add leading zeros
Z2 = t.cat([t.zeros(2,2),Z1],dim=1)
Z3 = Z2.unsqueeze(dim=1)
print(
	'ready for dilated causal' +
	f' conv:\n\t{Z3}'
)
Z4 = Y(Z3)
Z5 = Z4.permute(
	dims=[2,1,0]
).reshape(1,-1)
print(
	'dilated causal conv:\n' +
	f'\t{Z5.detach()}'
)
