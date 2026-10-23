from pyfoma import FST

wiki = 'est_latn_broad.tsv'

f = open(wiki,'r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

s1 = FST.re("$^rewrite(ó:o)")
s2 = FST.re("$^rewrite(ʋ:v)")
s3 = FST.re("$^rewrite((lːʲ):(lʲː))")
s4 = FST.re("$^rewrite((sːʲ):(sʲː))")
s5 = FST.re("$^rewrite((o̯̯):(o̯))")
s6 = FST.re("$^rewrite(g:(ɡ̊))")
s7 = FST.re("$^rewrite((ʒ̥):(ʒ̊))")

s = FST.re(
	"$s1 @ $s2 @ $s3 @ $s4 @ $s5 @ $s6 @ $s7",
	{'s1':s1,'s2':s2,'s3':s3,'s4':s4,'s5':s5,
	's6':s6,'s7':s7}
)

for line in lines:
	word,trans = line.split('\t')
	nt = list(s.generate(trans))[0]
	print(f'{word}\t{nt}\n')

