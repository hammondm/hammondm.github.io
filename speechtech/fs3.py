from pyfoma import FST

#all the transductions
hrule = FST.re("$^rewrite('' : H / #(t|k) _)")
prule = FST.re("$^rewrite((pp) : p)")
erule = FST.re("$^rewrite(e:'' / _ #)")
ae1rule = FST.re("$^rewrite((ae):e / _ . e #)")
ae2rule = FST.re("$^rewrite(a:(ae))")
yrule = FST.re("$^rewrite(y:i / _ #)")
eerule = FST.re("$^rewrite((ee):i)")
irule = FST.re("$^rewrite(i:I)")
c1rule = FST.re("$^rewrite(c:s / _ (i|e))")
c2rule = FST.re("$^rewrite(c:k)")

#compose everything together
allrules = FST.re(
	"$c1 @ $c2 @ $i @ $h @ $p @ $ae2 @ $ae1 @ $e @ $y @ $ee",
	{
		'h':hrule,
		'p':prule,
		'e':erule,
		'ae1':ae1rule,
		'ae2':ae2rule,
		'y':yrule,
		'ee':eerule,
		'i':irule,
		'c1':c1rule,
		'c2':c2rule
	}
)

#read in test words
f = open('words.txt','r')
ws = f.read()
f.close()

#apply compose transducer to all words
ws = ws.split('\n')
ws = ws[:-1]
for w in ws:
	res = list(allrules.generate(w))
	print(f'{w}: {res}')

