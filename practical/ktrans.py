
f = open('uni/kir.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

table = str.maketrans(
	"жиүыcэoбрмсчyᴙушpпaeгкөңлнзйтд",
	"ZiUIsEobrmsCuAuSrpaegkONlnzjtd"
)

for line in lines:
	bits = line.split('\t')
	b0 = bits[0].translate(table)
	b1 = bits[1].translate(table)
	print(f'{b0}\t{b1}\t{bits[2]}')

