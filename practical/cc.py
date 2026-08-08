from pyfoma import FST

pfx = '...'
wiki = 'wikiproncym.txt'
lets = 'letters.txt'

#read wiki file
f = open(pfx+wiki,'r')
t = f.read()
f.close()

#break into lines
words = t.split('\n')[:-1]

#remove multi-word items
extra = FST.re("~(.*' '.*'\t'.*)")

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#rule out voiced stop before voiceless stop
cc = FST.re(".*[bdg][ptk].*")

#compose
everything = FST.re(
	"$e @ $d @ $s @ $c",
	{'e':extra,'d':drop,'s':spaces,'c':cc}
)

for word in words:
	res = list(everything.generate(word))
	if len(res) > 0:
		print(word)

