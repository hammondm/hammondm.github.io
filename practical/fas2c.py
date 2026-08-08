from pyfoma import FST

wiki = 'wikipronfas.txt'
cons = 'fasconsonants.txt'

#read wiki file
f = open(wiki,'r')
t = f.read()
f.close()

#break into lines
words = t.split('\n')[:-1]

#read consonants file
f = open(cons,'r')
t = f.read()
f.close()

#break into lines
cons = t.split('\n')[:-1]
cons = FST.re('(' + '|'.join(cons) + ')')

#define a word as not beginning with 2 consonants
cc = FST.re('~($cons $cons .*)',{'cons':cons})

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#compose
everything = FST.re(
	"$drop @ $spaces @ $cc",
	{'drop':drop,'spaces':spaces,'cc':cc}
)

for word in words:
	res = list(everything.generate(word))
	if len(res) == 0:
		print(word)

