from matplotlib.patches import \
	Rectangle as rect
import matplotlib.pyplot as plt

#set of frequency/amplitude spikes
pairs = [(2,8),(4,4)]

fig = plt.figure()
ax = fig.add_subplot(111)
#define plot size
plt.xlim([0,40])
plt.ylim([0,12])
#plot the freq/amp spikes
for pair in pairs:
	print(pair)
	plt.vlines(
		pair[0],
		0,
		pair[1],
		color='blue'
	)
#add a rectangle
rect1 = rect((10,0),20,9,color='blue')
ax.add_patch(rect1)
plt.show()
