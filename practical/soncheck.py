import panphon.sonority as ps

s = ps.Sonority()

for x in "aijrmvfdt":
	print(f'{x}: {s.sonority(x)}')

