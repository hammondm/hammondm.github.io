from pyfoma import FST
import pfeatures as pf

#read in data
f = open('wikifasSTRESS.txt','r')
t = f.read()
f.close()
lines = t.split('\n')[:-1]

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#fix [g] problem
grule = FST.re("$^rewrite(g:ɡ)")

#eliminate syllable boundaries
dotrule = FST.re("$^rewrite('.':'')")

#fix t͡ʃ problem
chrule = FST.re("$^rewrite((tʃ):(t͡ʃ))")

#onset rule
onset = FST.re(
	"$sm? | $sm [mvrlɾw]",
	{'sm':pf.sylminus}
)

#simple syllables
syllable = FST.re(
	"$o $sp $sm{0,2}",
	{'sp':pf.sylplus,'sm':pf.sylminus,'o':onset}
)

#a word is one or more syllables
word = FST.re(
	"(ˈ? ($s):w)+",
	{'s':syllable}
)

#mark stress
stress = FST.re(
	"$^rewrite((ˈw):s)"
)

#pattern to search for
pat = FST.re(
	".*s.*w"
)

#put it all together
everything = FST.re(
	"$d @ $s @ $g @ $dot @ $ch @ $w @ $str @ $pat",
	{'d':drop,'s':spaces,'g':grule,'dot':dotrule,
	'ch':chrule,'w':word,'str':stress,'pat':pat}
)

for line in lines:
	res = set(everything.generate(line))
	if len(res) > 0:
		print(f'{line}\n\t{res}')

