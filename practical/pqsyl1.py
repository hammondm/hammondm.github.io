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
	"$sm* $sp $sm*",
	{'sp':pf.sylplus,'sm':pf.sylminus}
)

word = FST.re(
	"$s+",
	{'s':syllable}
)

everything = FST.re(
	"$d @ $s @ $g @ $w",
	{'d':drop,'s':spaces,'g':grule,'w':word}
)

print(f'arcs: {everything.arccount()}')
print(f'states: {len(everything.states)}')

