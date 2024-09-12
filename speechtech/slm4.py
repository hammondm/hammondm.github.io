from slm3 import myshortestpath

#example
fstx = '''0	0	a	b	2
0	1	a	a	2
0	1	b	b	3
1	1	b	c	2
1	2	b	b	3
1
2'''

#calculate path
dx,px = myshortestpath(fstx)
print('\nd:',dx)
print('p:',px)
