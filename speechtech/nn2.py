import numpy as np

#set variables
slope = 2
intercept = 7
r = .1
w = 3
b = 5
iterations = 10
samples = 100

#set random initial values
x = np.random.randint(-10,10,samples)

#go through initial values
d = []
ones = 0
#assign labels
for i in range(samples):
	if slope*x[i]+intercept > 0:
		lab = 1
		ones += 1
	else:
		lab = 0
	tup = (x[i],lab)
	d.append(tup)

print(ones)

print(f'target weight: {slope}')
print(f'target bias: {intercept}')

print(f'initial weight: {w}')
print(f'initial bias: {b}')

#iterate through data
for j in range(iterations):
	#adjust weight and bias accordingly
	for i in d:
		y = b + i[0]*w
		if y > 0:
			y = 1
		else:
			y = 0
		w += r * (i[1]-y) * i[0]
		b += r * (i[1]-y) * 1

print(f'later weight: {w}')
print(f'later bias: {b}')

#calculate successes
successes = 0
for item in d:
	res = item[0]*w + b
	if res > 0:
		res = 1
	else:
		res = 0
	if res == item[1]:
		successes += 1

print(f'success: {successes/samples}')

