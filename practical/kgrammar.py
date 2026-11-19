from pyfoma import FST

#read in data
f = open('kortho.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#collect all occurring tags
combos = set()
for line in lines:
	bits = line.split('\t')
	combos.add(bits[2])

#incorrectly mark all 2nd persons for FRML/INFM
two = FST.re("('FRML' ; 2)|(2 ; 'INFM')")

#person marking
person = FST.re(
	"1|$two|3",
	{'two':two}
)

#number
number = FST.re("'SG'|'PL'")

#tenses
tense = FST.re("'FUT'|'PRS'|'PST'")

#moods
mood = FST.re("'COND'|'HAB'|'PROG'|'PRF'")

#put them all together, overgenerating
tagset = FST.re(
	"V ; ($m ;)? $n ; $p ; $t",
	{'m':mood,
	'n':number,
	'p':person,
	't':tense}
)

#various repairs and filters
tagfix = FST.re(
	"$^rewrite(('SG' ; 'FRML'):('FRML' ; 'SG')) @ " +
	"$^rewrite(('COND' ; 'FRML'):('FRML' ; 'COND')) @ " +
	"~(.* ('PRF'|'HAB'|'PROG') ; 'PL' ; 'FRML' .*) @ " +
	"~(.* 'COND' .* ('PRS'|'PST') .*) @ " +
	"~(.* 'PROG' .* 'FUT' .*) @ " +
	"~('V' ; 'PL' ; 'FRML' ; 2 ; ('PST'|'PRS'|'FUT')) @ "
	"~('V' ; 'COND' ; 'PL' ; 'FRML' ; 2 ; 'FUT') @ "
	"~(.* ('PRF'|'HAB') .* ('FUT'|'PRS') .*)"
)

#generate the full set
wds = tagset.words()
gen = set()
for w in wds:
	tag = ''.join([x[0] for x in w[1]])
	#filter each item
	fixed = tagfix.generate(tag)
	for f in fixed:
		gen.add(f)

#check that there are no missing tags
for tag in combos:
	if tag not in gen:
		print(f'missing: {tag}')

#check that there are no extra tags
for tag in gen:
	if tag not in combos:
		print(f'extra: {tag}')

