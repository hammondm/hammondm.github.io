from pyfoma import FST
import pfeatures as pf

f = open('fas4.txt','r')
t = f.read()
f.close()

lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#fix [g] problem
grule = FST.re("$^rewrite(g:ɡ)")

#fix t͡ʃ problem
chrule = FST.re("$^rewrite((tʃ):(t͡ʃ))")

#onset rule
onset = FST.re("$sm? | $sm [mvrlɾw]",{'sm':pf.sylminus})

#simple syllables
syllable = FST.re(
	"$o $sp $sm{0,2}",
	{'sp':pf.sylplus,'sm':pf.sylminus,'o':onset}
)

#a word is one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions without vowels
exceptions = FST.re("b|x|z|r|n|ɹ|ɾ|p|q|β|ʒ|t͡ʃ|heɾtz|ɾekowɾd|shijaːr")

#put it all together
everything = FST.re(
	"$d @ $s @ $ch @ $g @ ($w|$e)",
	{'d':drop,'ch':chrule,'s':spaces,'g':grule,
	'w':word,'e':exceptions}
)

for line in lines:
	res = list(everything.generate(line))
	if len(res) != 1:
		print(line)

