from pyfoma import FST

wiki = 'kir_cyrl_broad.tsv'
lets = 'kletters.txt'
vows = 'kvowels.txt'

#read wiki file
f = open(wiki,'r')
t = f.read()
f.close()

#break into lines
words = t.split('\n')[:-1]

#read vowels file
f = open(vows,'r')
t = f.read()
f.close()

#break into lines
vowels = t.split('\n')[:-1]
vowels = FST.re('(' + '|'.join(vowels) + ')')

#read letters file
f = open(lets,'r')
t = f.read()
f.close()

#break into lines
letters = t.split('\n')[:-1]
letters = FST.re('(' + '|'.join(letters) + ')')

#any non-vowel is a consonant
cons = FST.re(
	"$letters - $vowels",
	{'letters':letters,'vowels':vowels}
)

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

