import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

def spec(pairs):
	#how tall is the plot
	mx = max([pair[0] for pair in pairs])
	plt.xlim(0,mx+1)
	#plot each frequency/amplitude pair
	for pair in pairs:
		print(pair)
		plt.vlines(pair[0],0,pair[1])
	plt.show()

#demo the function
spec([(2,2),(4,1)])
