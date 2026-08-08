from pyfoma import FST

pfx = '...'
wiki = 'wikiproncym.txt'
lets = 'letters.txt'
vows = 'vowels.txt'

#read wiki file
f = open(pfx+wiki,'r')
t = f.read()
f.close()

#break into lines
words = t.split('\n')[:-1]

#read letters file
f = open(lets,'r')
t = f.read()
f.close()

#break into lines
letters = t.split('\n')[:-1]

#read vowels file
f = open(vows,'r')
t = f.read()
f.close()

#break into lines
vowels = t.split('\n')[:-1]

#make letter and vowel FSAs
letter = FST.re('(' + '|'.join(letters) + ')')
vowel = FST.re('(' + '|'.join(vowels) + ')')

#define a word as having at least one vowel
wordshape = FST.re(
	'$letter* $vowel $letter*',
	{'letter':letter,'vowel':vowel}
)

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#compose
everything = FST.re(
	"$drop @ $spaces @ $wordshape",
	{'drop':drop,'spaces':spaces,'wordshape':wordshape}
)

for word in words:
	res = list(everything.generate(word))
	if len(res) == 0:
		print(word)

