from pyfoma import FST
import pfeatures as pf

#drop everything up to tab
drop = FST.re("$^rewrite((.*'\t'+):'',leftmost=True)")

#push letters together
spaces = FST.re("$^rewrite(' ':'')")

#fix [g] problem
grule = FST.re("$^rewrite(g:ɡ)")

#simple syllables
syllable = FST.re(
	"$sm{0,2} $sp $sm{0,3}",
	{'sp':pf.sylplus,'sm':pf.sylminus}
)

#a word is one or more syllables
word = FST.re(
	"$s+",
	{'s':syllable}
)

#exceptions without vowels
exceptions = FST.re("b|x|z|r|n|ɹ|ɾ|p|q|β|ʒ|tʃ")

#put it all together
everything = FST.re(
	"$d @ $s @ $g @ ($w|$e)",
	{'d':drop,'s':spaces,'g':grule,'w':word,'e':exceptions}
)

print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

